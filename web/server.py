"""FaultLens web server (FastAPI). Serves the console page and the JSON API.

Run from the project folder:
    python web/server.py            # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image  # noqa: E402

from faultlens import cardreader, coach, parts, partcheck, config, game, jobcards, knowledge, library, pattern, render, sensors, voice  # noqa: E402
from faultlens.checker import check as run_checker  # noqa: E402
from faultlens.agent import Evidence  # noqa: E402
from faultlens.media import draw_boxes, extract_keyframes  # noqa: E402
from faultlens.pipeline import locate_parts, match_patterns, run_round  # noqa: E402

app = FastAPI(title="FaultLens")
INDEX = Path(__file__).parent / "index.html"
library.LIBRARY.mkdir(exist_ok=True)
app.mount("/library", StaticFiles(directory=library.LIBRARY), name="library")


def _data_url(img: Image.Image, fmt: str = "PNG", max_side: int | None = None) -> str:
    img = img.convert("RGB")
    if max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 82} if fmt == "JPEG" else {}))
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _signal_views(ev: Evidence) -> dict:
    """Pictures of each signal for the console."""
    views = {}
    if ev.thermal_indoor is not None:
        views["thermal_indoor"] = _data_url(render.thermal_view(ev.thermal_indoor))
    if ev.thermal_outdoor is not None:
        views["thermal_outdoor"] = _data_url(render.thermal_view(ev.thermal_outdoor))
    if ev.sound is not None:
        views["spectrogram"] = _data_url(render.spectrogram(ev.sound))
    return views


def _load_sample_into(ev: Evidence, cond: str, sample: str) -> None:
    d = library.sample_dir(cond, sample)
    ev.sound = render.read_wav((d / "sound.wav").read_bytes())
    ev.thermal_indoor = render.read_thermal_png((d / "thermal_indoor.png").read_bytes())
    ev.thermal_outdoor = render.read_thermal_png((d / "thermal_outdoor.png").read_bytes())
    ev.sensor_rows = sensors.from_csv((d / "sensors.csv").read_text())


def _route_upload(ev: Evidence, name: str, ctype: str, data: bytes, log: list) -> None:
    """Work out what an uploaded file is and put it in the right place."""
    lower = name.lower()
    if lower.endswith(".wav") or ctype in ("audio/wav", "audio/x-wav", "audio/wave"):
        ev.sound = render.read_wav(data)
        log.append({"file": name, "as": "Sound (outdoor unit)"})
    elif lower.endswith(".csv"):
        ev.sensor_rows = sensors.from_csv(data.decode(errors="replace"))
        log.append({"file": name, "as": "Sensor log"})
    elif ctype.startswith("video") or lower.endswith((".mp4", ".mov", ".webm")):
        with tempfile.NamedTemporaryFile(suffix=Path(lower).suffix or ".mp4", delete=False) as tmp:
            tmp.write(data)
        ev.frames.extend(extract_keyframes(tmp.name, config.KEYFRAMES_PER_VIDEO))
        log.append({"file": name, "as": "Video (keyframes)"})
    else:
        try:
            temps = render.read_thermal_png(data)
        except ValueError:
            temps = None
        except Exception:
            raise HTTPException(400, f"Could not read {name}.")
        if temps is not None:
            if "indoor" in lower:
                ev.thermal_indoor = temps
                log.append({"file": name, "as": "Thermal image, indoor coil"})
            else:
                ev.thermal_outdoor = temps
                log.append({"file": name, "as": "Thermal image, outdoor unit"
                            + ("" if "outdoor" in lower else " (name has no 'indoor', so treated as outdoor)")})
        else:
            ev.frames.append(Image.open(io.BytesIO(data)).convert("RGB"))
            log.append({"file": name, "as": "Photo"})


# ------------------------------------------------------------------ pages and info

@app.get("/")
def index():
    return FileResponse(INDEX)


@app.get("/api/profile")
def profile():
    p = knowledge.load()
    return {
        "equipment": p["equipment"],
        "faults": knowledge.fault_menu(),
        "mock": config.MOCK,
        "pattern_model": pattern.get_model() is not None,
        "models": {"agent": config.AGENT_MODEL, "sam": config.SAM_MODEL, "voice": config.VOICE_MODEL,
                   "checker": config.CHECKER_MODEL or config.AGENT_MODEL},
    }


@app.get("/api/library")
def library_listing():
    return {"conditions": library.listing()}


@app.get("/api/library/{cond}/{sample}")
def library_sample(cond: str, sample: str):
    try:
        ev = Evidence()
        _load_sample_into(ev, cond, sample)
    except FileNotFoundError:
        raise HTTPException(404, f"No sample {cond}/{sample}. Build the library: python -m faultlens.library")
    note = match_patterns(ev)
    info = json.loads((library.sample_dir(cond, sample) / "info.json").read_text())
    return {"info": info, "views": _signal_views(ev), "sensor_summary": ev.sensor_summary(),
            "pattern_match": ev.pattern_match, "pattern_note": note}


@app.get("/api/pattern-report")
def pattern_report():
    if not pattern.REPORT_PATH.exists():
        raise HTTPException(404, "No report yet. Train the model: python -m faultlens.pattern")
    return json.loads(pattern.REPORT_PATH.read_text())


@app.get("/api/vision-report")
def vision_report():
    meta, open_ = ROOT / "models" / "vision_report.json", ROOT / "models" / "vision_report_open.json"
    if not meta.exists() and not open_.exists():
        raise HTTPException(404, "Not measured yet. Run: python -m eval.vision_eval")
    return {"meta": json.loads(meta.read_text()) if meta.exists() else None,
            "open": json.loads(open_.read_text()) if open_.exists() else None}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        return {"text": voice.transcribe(await audio.read())}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # surface the real API error to the page
        raise HTTPException(502, f"Transcription failed: {e}")


# ------------------------------------------------------------------ investigate

@app.post("/api/investigate")
async def investigate(
    files: list[UploadFile] = File(default=[]),
    sensor_csv: str = Form(""),
    library_sample: str = Form(""),
    notes: str = Form("[]"),
    find_parts: bool = Form(True),
    use_checker: bool = Form(True),
):
    ev = Evidence(notes=[n for n in json.loads(notes or "[]") if n.strip()])
    received: list[dict] = []

    if library_sample:
        cond, _, sample = library_sample.partition("/")
        try:
            _load_sample_into(ev, cond, sample)
        except FileNotFoundError:
            raise HTTPException(404, f"No library sample {library_sample}.")
        received.append({"file": library_sample, "as": "Library sample: sound, both thermal images, sensor log"})
    if sensor_csv.strip():
        ev.sensor_rows = sensors.from_csv(sensor_csv)
        received.append({"file": "pasted text", "as": "Sensor log"})
    for f in files:
        data = await f.read()
        if data:
            try:
                _route_upload(ev, f.filename or "upload", f.content_type or "", data, received)
            except ValueError as e:
                raise HTTPException(400, f"{f.filename}: {e}")

    has_signal = ev.sound is not None or ev.thermal_indoor is not None or ev.thermal_outdoor is not None
    if not (ev.sensor_rows or ev.notes or ev.frames or has_signal):
        raise HTTPException(400, "Add at least one piece of evidence: sound, a thermal image, a sensor log, "
                                 "a photo or a note.")

    timings: dict = {}
    try:
        if find_parts and ev.frames:
            t = time.time()
            locate_parts(ev)
            timings["parts"] = round(time.time() - t, 2)
        result = run_round(ev, use_checker=use_checker, timings=timings)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Investigation failed: {e}")

    photos = []
    for i, frame in enumerate(ev.frames):
        dets = [d for d in ev.detections if d.get("frame_index") == i]
        photos.append({"image": _data_url(draw_boxes(frame, dets) if dets else frame, "JPEG", 900),
                       "parts": sorted({d["label"] for d in dets})})

    titles = {f["fault_id"]: f["title"] for f in knowledge.fault_menu()}
    for h in result["assessment"]["hypotheses"]:
        h["title"] = titles.get(h["fault_id"], h["fault_id"])

    return {**result, "photos": photos, "views": _signal_views(ev), "received": received,
            "sensor_summary": ev.sensor_summary(), "notes_used": ev.notes, "titles": titles}


# ------------------------------------------------------------------ job cards

@app.get("/api/parts")
def part_list():
    titles = {f["fault_id"]: f["title"] for f in knowledge.fault_menu()}
    _, outlined_by = parts.boxes()
    items = parts.listing()
    for it in items:
        it["fault_titles"] = [titles.get(c, c) for c in it["faults"]]
    return {"parts": items, "outlined_by": outlined_by,
            "by_fault": {c: parts.parts_for(c) for c in parts.FAULT_PARTS},
            "diagram": "/library/diagram/split_ac_exploded.png"}


@app.get("/api/diagram.jpg")
def diagram(highlight: str = ""):
    nums = [int(x) for x in highlight.split(",") if x.strip().isdigit() and int(x) in parts.PARTS]
    buf = io.BytesIO()
    parts.highlight(nums).save(buf, format="JPEG", quality=85)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/partcheck")
async def part_check(file: UploadFile | None = File(default=None), library_url: str = Form(""),
                     part_no: str = Form(""), complaint: str = Form(""), job_id: str = Form("")):
    lib_path = None
    if library_url:
        rel = library_url.split("/library/", 1)[-1]
        lib_path = (library.LIBRARY / rel).resolve()
        if library.LIBRARY.resolve() not in lib_path.parents or not lib_path.exists():
            raise HTTPException(400, "That library photo does not exist.")
        img = Image.open(lib_path).convert("RGB")
    elif file is not None:
        try:
            img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"Could not open that photo: {e}")
    else:
        raise HTTPException(400, "Upload a photo of the part.")
    if job_id and job_id in game.JOBS and not complaint:
        complaint = game.JOBS[job_id].complaint
    n = int(part_no) if part_no.strip().isdigit() and int(part_no) in parts.PARTS else None
    try:
        res = partcheck.check(img, n, complaint, lib_path)
    except Exception as e:
        raise HTTPException(502, f"Part check failed: {e}")
    res["photo"] = _data_url(img, "JPEG", 900)
    res["outlined"] = _data_url(res["outlined"], "JPEG", 900) if res["outlined"] is not None else None
    res["healthy_reference"] = f"/library/parts/{parts.PARTS[res['part_no']][2]}" if parts.PARTS[res["part_no"]][2] else None
    res["diagram"] = f"/api/diagram.jpg?highlight={res['part_no']}"
    return res


@app.get("/api/jobcards")
def jobcard_list():
    titles = {f["fault_id"]: f["title"] for f in knowledge.fault_menu()}
    cards = jobcards.listing()
    for c in cards:
        c["title"] = titles.get(c["condition"], c["condition"])
    return {"cards": cards, "fields": [{"key": k, "label": jobcards.LABELS[k]} for k in jobcards.FIELDS]}


@app.post("/api/jobcard/read")
async def jobcard_read(file: UploadFile | None = File(default=None), card_id: str = Form("")):
    if card_id:
        img = jobcards.image_for(card_id)
        if img is None:
            raise HTTPException(404, f"No job card {card_id}.")
    elif file is not None:
        try:
            img = cardreader.load_upload(await file.read(), file.filename or "")
        except Exception as e:
            raise HTTPException(400, f"Could not open that file: {e}")
    else:
        raise HTTPException(400, "Choose a library card or upload a PDF or photo of a job card.")
    try:
        res = cardreader.read(img, card_id or None)
    except Exception as e:
        raise HTTPException(502, f"Reading failed: {e}")
    truth = jobcards.truth_for(card_id) if card_id else None
    res["score"] = cardreader.score(res["fields"], truth) if truth else None
    res["image"] = _data_url(img, "JPEG", 1400)
    return res


# ------------------------------------------------------------------ the job simulator

class NewJob(BaseModel):
    condition: str = ""      # empty = mystery job
    seed: int | None = None


class Act(BaseModel):
    action: str


class Power(BaseModel):
    on: bool


class Ask(BaseModel):
    question: str = ""


class Conclude(BaseModel):
    choice: str


def _job(job_id: str) -> game.Job:
    try:
        return game.get(job_id)
    except KeyError:
        raise HTTPException(404, "That job has expired. Start a new one.")


def _job_view(job: game.Job) -> dict:
    return {"id": job.id, "complaint": job.complaint, "practice": job.practice, "power_on": job.power_on,
            "condition": job.condition if job.practice else None,
            "actions": game.action_menu(job), "steps": job.steps, "blocked": len(job.blocked),
            "faults": knowledge.fault_menu()}


@app.post("/api/game/new")
def game_new(body: NewJob):
    try:
        job = game.new_job(body.condition or None, seed=body.seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _job_view(job)


@app.post("/api/game/{job_id}/action")
def game_action(job_id: str, body: Act):
    job = _job(job_id)
    try:
        found = game.perform(job, body.action)
    except ValueError as e:
        raise HTTPException(400, str(e))
    views = {}
    a = body.action
    if not found["blocked"]:
        if a == "listen_outdoor":
            views["spectrogram"] = _data_url(render.spectrogram(job.signals["sound"]))
            views["sound_url"] = f"/api/game/{job.id}/sound.wav"
        elif a == "thermal_indoor":
            views["thermal"] = _data_url(render.thermal_view(job.signals["thermal_indoor"]))
        elif a == "thermal_outdoor":
            views["thermal"] = _data_url(render.thermal_view(job.signals["thermal_outdoor"]))
        elif a == "read_sensors":
            views["sensor_summary"] = sensors.summarize(job.signals["sensors"])
    return {"found": found, "views": views, "pattern_match": coach.pattern_for(job), "job": _job_view(job)}


@app.get("/api/game/{job_id}/sound.wav")
def game_sound(job_id: str):
    job = _job(job_id)
    if "sound" not in job.signals:
        raise HTTPException(404, "No sound recorded yet on this job.")
    return Response(render.write_wav(job.signals["sound"]), media_type="audio/wav")


@app.post("/api/game/{job_id}/power")
def game_power(job_id: str, body: Power):
    job = _job(job_id)
    msg = game.set_power(job, body.on)
    return {"message": msg, "job": _job_view(job)}


@app.post("/api/game/{job_id}/advise")
def game_advise(job_id: str, body: Ask):
    job = _job(job_id)
    t = time.time()
    advice = coach.advise(job, body.question.strip() or None)
    advice["seconds"] = round(time.time() - t, 1)
    titles = {f["fault_id"]: f["title"] for f in knowledge.fault_menu()}
    for h in advice["hypotheses"]:
        h["title"] = titles.get(h["fault_id"], h["fault_id"])
    nxt = advice.get("next_action_id")
    advice["next_title"] = (game.ACTIONS[nxt][1] if nxt in game.ACTIONS else
                            {"power_off": "Isolate the power", "power_on": "Restore power",
                             "conclude": "Make the diagnosis"}.get(nxt, ""))
    return advice


@app.post("/api/game/{job_id}/conclude")
def game_conclude(job_id: str, body: Conclude):
    job = _job(job_id)
    if job.concluded:
        return job.concluded
    adv = job.last_advice or {}
    coach_choice = adv.get("final_fault_id")
    verdict = None
    if adv.get("hypotheses"):
        # The second model reviews FaultLens's final call against the same evidence and images.
        ev = Evidence(notes=[f"{s['title']}: {s['finding']}" for s in job.steps])
        ev.sound = job.signals.get("sound")
        ev.thermal_indoor = job.signals.get("thermal_indoor")
        ev.thermal_outdoor = job.signals.get("thermal_outdoor")
        ev.sensor_rows = job.signals.get("sensors", [])
        ev.pattern_match = coach.pattern_for(job)
        assessment = {"status": "confident" if adv.get("ready_to_conclude") else "need_more_evidence",
                      "hypotheses": adv["hypotheses"], "summary": adv.get("message", "")}
        try:
            verdict = run_checker(ev, assessment)
        except Exception as e:
            verdict = {"verdict": "unavailable", "reasons": [str(e)], "checker_model": ""}
    result = game.score(job, body.choice, coach_choice)
    titles = {f["fault_id"]: f["title"] for f in knowledge.fault_menu()}
    result.update(player_title=titles.get(body.choice, body.choice),
                  coach_title=titles.get(coach_choice, coach_choice) if coach_choice else None,
                  checker=verdict, fix=knowledge.fault_guide(job.condition)["fix"],
                  faulty_parts=[{"no": n, "name": parts.PARTS[n][0]} for n in parts.parts_for(job.condition, job.variant)],
                  outlined_by=parts.boxes()[1],
                  steps=job.steps)
    job.concluded = result
    return result


@app.post("/api/eval")
def run_eval(limit: int = 12, modes: str = "rules,agent_sensors,agent_full"):
    from eval.run_eval import evaluate, pick_cases

    files = pick_cases(limit)
    results, report = evaluate(files, modes.split(","))
    return {"report": report, "results": results}


if __name__ == "__main__":
    if pattern.get_model() is None:
        print("Note: pattern model not trained yet. Run: python -m faultlens.pattern")
    if not any(library.LIBRARY.iterdir()):
        print("Note: reference library is empty. Run: python -m faultlens.library")
    print("FaultLens running at http://127.0.0.1:8000" + ("  (offline mode)" if config.MOCK else ""))
    uvicorn.run(app, host="127.0.0.1", port=8000)
