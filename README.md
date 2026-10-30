# FaultLens

A diagnostic agent that helps technicians work out why a split air conditioner has failed, and what to check next.

The technician records what they can at the unit: the sound of the outdoor unit, thermal images of the indoor coil and the outdoor unit, a sensor log, photos and spoken notes. FaultLens compares the signals with a library of fault signatures, combines them into one ranked answer, explains which measurements point where, and recommends the next check. When the signals disagree or the evidence is thin, it asks for one specific observation instead of guessing. A second model reviews every diagnosis, and fixed safety rules gate any step that needs the power isolated or refrigerant handled.

Built for the Meta Global AI Developer Hackathon 2026 (Indie track).

## Play a job

The **Play a job** view is a service-job simulator. A customer describes the problem; the fault is hidden.
The technician works through it with FaultLens, one check at a time: listen to the outdoor unit, take
thermal images, log readings, inspect parts, measure components. Every check reveals what a technician
would find on that unit, and every job is generated fresh, so sound, temperatures and readings differ each time.

- **FaultLens (Muse Spark) is the master diagnostician.** After each check it looks at everything found so far,
  including the thermal images and the sound spectrogram, ranks the likely causes, and recommends the single
  next check that best separates them. You can follow it, choose your own check, or ask it a question.
- **Safety is part of the game.** Checks that need the unit running are blocked while power is off; checks
  that open the electrical compartment are stopped until the power is isolated, and each unsafe attempt costs points.
- **The live fault meter** shows FaultLens's ranking next to the pattern model's ranking from the signals collected.
- **The reveal** shows the hidden fault, your call, FaultLens's call, whether the second model (Muse Spark Open)
  agreed with FaultLens, the fix, and a score.

Photos and the exploded-view diagram are added in `library/` (see `library/PHOTOS.md`).

## Try it in the browser

The **Explore faults** view is a working simulation of every fault. Each fault has the same four signals:

| Signal | File | What it shows |
|---|---|---|
| Sound at the outdoor unit | `sound.wav` | Compressor, fan and electrical hum; trips, clicks and hiss |
| Thermal, indoor coil | `thermal_indoor.png` | Coil temperature, ice patches, starved coil |
| Thermal, outdoor unit | `thermal_outdoor.png` | Coil, compressor and capacitor temperatures |
| Sensor log | `sensors.csv` | Air temperatures, pressures and currents, one row a minute |

Pick a fault, listen to it, compare the thermal images, then press **Investigate this sample**. Or download the files and upload them under **Investigate**, which is exactly what a technician does in the field with their own recordings.

## Anomaly lab: inject a fault, then find it

Pick a signal (6 hours of sensor readings, or 60 seconds of sound) and a fault to inject. A healthy unit runs; partway through, the fault creeps in, stays, and is then repaired. FaultLens's own pattern model slides a window along the signal and labels every stretch, and the chart shows what it called, with its confidence, above the true fault stretch. It uses the trained model only: no API calls.

An alarm is raised only when the model is sure enough for two windows in a row (80% for sensors, 70% for sound). Those settings were tuned on separate runs (seeds 5000-5007), then measured on runs never used for tuning (`models/anomaly_report.json`, 20 runs per fault):

| Injected | Sensor log: right fault | Sensor log: caught after (median) | Sound: right fault | Sound: caught after (median) |
|---|---|---|---|---|
| Clogged filter | 20/20 | 18.5 min | 10/20 | 4.5 s |
| Outdoor fan failure | 20/20 | 41.5 min | 20/20 | 5.5 s |
| Low refrigerant | 20/20 | 11.8 min | 20/20 | 8.7 s |
| Dirty outdoor coil | 20/20 | 15.6 min | 20/20 | 7.5 s |
| Failed run capacitor | 20/20 | 34.8 min | 20/20 | 3.2 s |
| Faulty room sensor | 20/20 | 11.2 min | 20/20 | 3.2 s |

About 1 run in 10 still has a false alarm. Each run is scored on four things, shown under the chart: right fault found, how much of the true stretch it covered, how late it caught it (a fault that starts mildly is hard), and false alarms while the unit was healthy. Two more are reported apart, because they are not false alarms: naming the wrong fault while it was still mild (a slowing fan looks like a dirty coil), and being slow to clear after the repair (each window still holds part of the fault).

After a run, **Ask FaultLens to explain** sends Muse Spark the run's data (the true stretch, every window's answer and confidence, the score, and the key reading or the spectrogram). It explains in plain words what changed, when and why the model caught it, and any delay or wrong label. One call, using only numbers from the run.

Sound alone misses half the clogged filters: a blocked indoor filter barely changes how the outdoor unit sounds. That is why FaultLens combines signals. Remeasure with your trained model: `python -c "from faultlens import anomaly; anomaly.measure(20)"` (about 3 minutes).

## Severity: how far a fault has developed

Every simulated unit has a severity, from early (under 45%) through developing (45-80%) to advanced (80-100%). It changes everything about that unit: the readings, the sound, both thermal images, the alarm log and the photos. FaultLens is never told the severity; it has to find the fault anyway, and early faults are the hard ones (98% overall, 94% on mild faults).

Every sign that changes as a fault develops has three photos, early, developing and advanced: frost on the indoor coil, frost at the coil inlet, an oily flare nut, a scorching fan-motor winding, a swelling fan or run capacitor, the capacitor meter reading (30.0, 24.9 and 18.2 uF on a 35 uF capacitor), a cracking sensor wire and the display against a thermometer (22, 20 and 18 C against 27 C). The game's findings use the same level, so the words and the photo always agree. In **Check a part**, Muse Spark also matches a photo to the closest of these levels ("How far has it gone?"); a library photo counts as right only when the level is exactly right.

The app shows it plainly: a bar with the sample's severity on it, each reading marked within normal, slightly out or far out against its normal range, and photos that match. Gradual wear (filter dust, coil dirt, fan-hub rust, pipe corrosion) shows the matching wear stage; signs that only appear once a fault is bad enough (frost on the coil) are left out below that point, with the reason; sudden failures (a swollen capacitor, a burnt winding) look the same at any severity and are marked so.

## Alarm history

Every unit also has its own fault log: the last 7 days of error codes, as a technician reads them from the display. The codes are a small demo table (each manufacturer uses its own) and the log is simulated from the fault, like the other signals.

The log is indirect evidence, as on a real unit. **E4** (outdoor too hot) fits a stopped outdoor fan and a dirty outdoor coil alike. A refrigerant leak only trips **E6** once it is bad. A room sensor that has slipped onto the cold coil reads a real but wrong temperature, so it leaves no code at all. FaultLens is told to treat the log as what the unit protected itself from, not as the answer.

It appears in Explore (with `alarms.csv` to download), in investigations (upload any CSV with a `code` column), in the job as **Read the unit's fault history**, and in the test set.

## Parts and photos

**Explore faults → Parts of the unit** shows a numbered exploded view of the split AC and a photo of each healthy part. Each fault page outlines its faulty part on the exploded view and shows the healthy part next to the damaged one: a burnt fan-motor winding, a rusted fan hub, a swollen capacitor, an oily flare nut, frost on the coil.

In a job, the specific cause is picked per unit (for example, the outdoor fan has stopped because of a seized bearing, a burnt motor winding or a failed capacitor), and the findings and photos match that cause. The reveal outlines the faulty part.

The outlines come from **SAM 3.1** once you run `python -m faultlens.parts --sam` (it saves `library/diagram/parts_sam.json`). Until then, boxes drawn by hand are used, and the app says which. SAM is given each part's name and returns every match; FaultLens keeps the one on the right part, which matters on a diagram with two capacitors.

The exploded view and all part and fault photos are generated with ChatGPT, and labelled as generated wherever they appear.

## Check a part from a photo

The technician photographs a part. **Check a part** sends Muse Spark the photo together with the reference photo of that part when healthy and the reference photos of it damaged, and gets back:

- which part it is (if the technician is not sure, a first step identifies it from a numbered sheet of all the parts),
- healthy, damaged or unclear, and what it sees,
- what to do: replace, clean, no action, or check further,
- whether the damage **causes** a fault or is only a **symptom** of one elsewhere (frost on the indoor coil is caused by a blocked filter or low refrigerant, not by the coil), and which fault,
- one check that would confirm it.

SAM 3.1 then outlines the part in the photo. In a job, every part photo has a **Check this part** button.

When a library photo is checked, that exact photo is left out of the references, so the model cannot match it to itself, and the known answer is shown beside the result.

**Measured** on 15 library photos (wear stages, severity levels, fault photos and healthy parts), with Muse Spark identifying the part itself and each photo and its copies left out of the references (`python -m eval.partcheck_eval`):

| | Result |
|---|---|
| Right part | 15 of 15 |
| Right condition (healthy or damaged) | 13 of 15 |
| Wear stage within one | 4 of 4 (all exactly right) |
| Severity level exact | 7 of 9 |
| Per check | about 16 s, about 10,000 tokens, about 2 US cents |

The misses: an oily flare nut at "developing" was called "advanced", and an indoor coil inlet at "early" (cold and sweating, no frost yet) was called healthy. An air filter at wear stage 2 ("light dust") was called healthy; it got the stage exactly right, but it counts as a condition miss because the library labels every stage above clean as damaged. These are generated photos, so this measures whether the comparison works, not accuracy on real units.

## Preventive maintenance: how long has it got?

Some parts wear gradually and visibly: the air filter (dust), the outdoor coil (dirt), the outdoor fan hub (rust) and the copper pipes (corrosion). For these, the library holds reference photos of each stage of wear, clean to worst (`library/wear/`). Check a part shows Muse Spark those stages, it says which stage the technician's photo matches, and FaultLens gives that stage's typical remaining life and the preventive action: nothing yet, plan it for the next service, schedule it within weeks, or do it on this visit.

The remaining life comes from a table (`faultlens/knowledge/split_ac_wear.yaml`), not from the model, and the app says so: it is a typical range for a split AC in normal use, not a measurement of that unit. Early stages count as normal ageing, not a fault. Capacitors and motor windings fail suddenly, so they have no wear scale.

When a stage photo from the library is checked, that stage's picture (and any copy of it) is left out, so FaultLens has to place it between its neighbours; it counts as right when it lands within one stage.

## Job cards: reading handwriting

Service technicians still write job cards by hand. The **Job cards** view holds 14 handwritten cards from earlier visits, two per condition, as PDF and image. Pick one, or upload your own PDF or photo, and FaultLens:

1. shows the scanned card,
2. reads all the writing on it,
3. puts the values into named fields (job number, date, complaint, pressures, work done, parts, remarks...) and shows them as a table and as JSON.

For the library cards the true contents are known, so each field is marked right or wrong. In a job, **Read the last job card** brings the previous visit into the investigation. The coach treats it as history, not proof.

The cards are generated (`python -m faultlens.jobcards`) with handwriting fonts (Caveat, Kalam, Nanum Pen Script, Homemade Apple, all under the SIL Open Font License) and scan effects. Customers and the company "Coolfix Home Services (demo)" are fictional.

## How it uses Meta's models

| Model | Job in FaultLens |
|---|---|
| **Muse Spark 1.3** | The investigation agent. Looks at the thermal images and the sound spectrogram itself, reads the pattern match, sensor log and notes, calls tools for fault guides, and submits a ranked assessment with evidence for and against each cause. |
| **SAM 3.1** | Finds named parts (filter, coil, fan, capacitor) in photos so the evidence can point at them. |
| **Muse Voice Transcribe** | Turns the technician's spoken notes into text. |
| **Muse Spark Open** (on Baseten) | The independent checker that accepts or rejects each diagnosis. It sees the same thermal images and **listens to the actual sound recording** of the outdoor unit, not just its spectrogram. |
| **Muse Spark Open** (on Fireworks) | Reads handwritten job cards into organised fields. Falls back to Muse Spark 1.3 if no Fireworks key is set. |

## The pattern model

One classifier per signal, combined by adding their log-probabilities, so the answer works with any subset of signals the technician uploads. Each signal's result comes with the measurements that drove it, in plain words ("Compressor, degrees above outside air: 48 (healthy about 27)").

Results on separately generated test units the model never saw (`models/pattern_report.json`):

| Signals used | Correct, all severities | Correct, mild faults (15-35%) |
|---|---|---|
| Sound only | 82% | 65% |
| Thermal, indoor coil only | 55% | 46% |
| Thermal, outdoor unit only | 83% | 70% |
| Sensor log only | 90% | 74% |
| **All four combined** | **98%** | **94%** |

No single signal is reliable on its own; combined, they are, and accuracy stays above 94% with any one signal missing. The **How it was tested** view shows the full report and where the model still gets confused.

**Honest confidence.** Each signal's model was overconfident on its own: it said 99% when it was right far less often. After training, its confidence is adjusted on a separate set of units, so that when the model says "80% sure" it is right about 80% of the time. The combined answer averages 98% confidence for 98% accuracy (95% for 94% on mild faults). This matters because the agent and the checker weigh the pattern match against what they see themselves.

Training and testing take about 2 minutes on a laptop (`python -m faultlens.pattern`).

## Public documentation

Each fault guide links to the public sources that back it, shown on the fault page, in the reveal and given to the agent with the guide (`faultlens/knowledge/split_ac_sources.yaml`):

- U.S. Department of Energy, [Air Conditioner Maintenance](https://www.energy.gov/energysaver/maintaining-your-air-conditioner)
- ENERGY STAR, [Maintenance Checklist](https://www.energystar.gov/saveathome/heating-cooling/maintenance-checklist)
- U.S. EPA, [Section 608 Technician Certification](https://www.epa.gov/section608/section-608-technician-certification-0) (refrigerant work by certified technicians)
- OSHA, [29 CFR 1910.147, lockout/tagout](https://www.law.cornell.edu/cfr/text/29/1910.147) (power isolation, discharging capacitors)

Each link says what that source actually covers. Where no public source covers a step (measuring a run capacitor against its rating, testing the room sensor's resistance, spinning the fan blade by hand), the guide says it comes from general service practice.

## Data: what is simulated and what is not

- **Sound, thermal images and sensor logs are simulated.** They are generated by `faultlens/simulate.py` and `faultlens/sensors.py` from simple physical models of how each fault changes the unit (for example: a stopped outdoor fan removes the blade tone, heats the coil from the top down and raises head pressure). Every unit differs: severity, outdoor temperature, installation, camera calibration and background noise all vary. Every simulated file is labelled "Simulated" in the app and in its `info.json`.
- **Job cards are generated**, with fictional customers. See "Job cards" above.
- **Photos** go in `library/<fault>/photos/`, with the source of each recorded in `photos.json` as `real-photo`, `chatgpt` or `muse-image`. The app shows that label on every photo.
- The fault guide in `faultlens/knowledge/split_ac.yaml` was written for this project from general service practice. No manual text is copied.

Because the model is trained on simulated signatures, its accuracy shows that the method works and that combining signals helps. It does not show field accuracy. The agent is told this and treats the pattern match as one piece of evidence.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env   -> then add your keys
python -m faultlens.pattern     # trains the pattern model (a few minutes)
python web/server.py            # open http://127.0.0.1:8000
```

The reference library is included. To rebuild or enlarge it: `python -m faultlens.library --per-condition 5`.

Offline mode with canned agent answers (the pattern model still runs): set `FAULTLENS_MOCK=1` in `.env`.

## Evaluation

**How well conclusions cite their evidence.** Every piece of evidence FaultLens gives, for and against each cause, is checked automatically (`eval/citations.py`): every number it quotes must appear in what it was given, it must not quote a sensor channel that was held back, describe a picture it was not shown, or name an alarm code that is not in the log. Saying that something is missing counts as correct. The test set reports the share of evidence that passes, next to accuracy and how often FaultLens asked for more evidence.


```bash
python -m faultlens.pattern               # pattern model: train + test, report in models/
python -m eval.make_cases                 # 28 agent test cases, half mild with a key sensor missing
python -m eval.run_eval --limit 14        # agent vs rule baseline; report in data/runs/
```

## Project layout

```
web/server.py              Web server (FastAPI): page + JSON API
web/index.html             The console: play a job, explore faults, job cards, investigate, how it was tested
smoke_test.py              First-call test for the Meta API key
faultlens/
  simulate.py              Simulated sound and thermal signatures per fault
  sensors.py               Simulated sensor logs, summaries, rule baseline
  features.py              Measurements extracted from each signal, with plain-language names
  pattern.py               Per-signal classifiers, fusion, training and testing
  library.py               Builds and lists the reference library
  render.py                Thermal false colour, spectrograms, file reading and writing
  game.py                  The service-job simulator: hidden fault, checks, findings, safety, score
  coach.py                 The master diagnostician: Muse Spark advising one step at a time
  agent.py                 The tool-using investigation agent (Muse Spark)
  checker.py               Independent checker (Muse Spark Open)
  safety.py                Fixed safety rules applied after the model
  pipeline.py              One full round: parts, patterns, agent, safety, checker
  partcheck.py             Check a part from a photo against healthy and damaged references
  anomaly.py               Anomaly lab: inject a fault into a healthy signal, find it with a sliding window
  alarms.py                The unit's fault log (demo error codes), simulated per fault
  wear.py                  Wear stages and typical remaining life for preventive maintenance
  parts.py                 Exploded view, parts list, outlines (SAM 3.1 or by hand)
  jobcards.py              Generates the handwritten job cards (PDF, image, true contents)
  cardreader.py            Reads a job card into fields (Muse Spark Open on Fireworks)
  fonts/                   Handwriting fonts and their licences (SIL OFL)
  sam.py, voice.py         SAM 3.1 and Muse Voice Transcribe
  knowledge/split_ac.yaml  Faults, checks, normal ranges, safety rules
library/                   Reference library, one folder per condition; library/jobcards/ holds the cards
models/                    Pattern model report (the model itself is rebuilt locally)
eval/                      Agent test set, runner and evidence-citation check
```

## Built during the hackathon (Oct 26 - Nov 4, 2026)

Everything in this repository was written during the hackathon. The commit history shows the timeline.
