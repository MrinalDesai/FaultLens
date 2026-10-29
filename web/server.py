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
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image  # noqa: E402

from faultlens import config, knowledge, library, pattern, render, sensors, voice  # noqa: E402
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
