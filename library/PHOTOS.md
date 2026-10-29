# Images for FaultLens

Three stages. Each stage uses the previous stage's images as reference, so parts look the same everywhere.
No brand names or logos on any image. `source` in `photos.json` is `real-photo`, `chatgpt` or `muse-image`.

## Stage 1: exploded view (1 image)

- [ ] `library/diagram/split_ac_exploded.png`: fixed-speed split AC, indoor and outdoor units, parts numbered 1-18 as below.

## Stage 2: healthy parts, `library/parts/` (generate the starred ones first)

| # | Part | File | Used by fault |
|---|---|---|---|
| 1 | Front panel and grille | `p01_front_panel.png` | - |
| 2 ★ | Air filters | `p02_air_filter.png` | No fault, Clogged filter |
| 3 ★ | Evaporator (indoor) coil | `p03_evaporator_coil.png` | No fault, Clogged filter, Low refrigerant |
| 4 | Blower wheel | `p04_blower_wheel.png` | - |
| 5 | Indoor fan motor | `p05_indoor_fan_motor.png` | - |
| 6 ★ | Room temperature sensor | `p06_room_thermistor.png` | Faulty room sensor |
| 7 | Indoor control board | `p07_indoor_board.png` | - |
| 8 | Drain pan and hose | `p08_drain_pan.png` | - |
| 9 ★ | Display and remote | `p09_display_remote.png` | Faulty room sensor |
| 10 ★ | Condenser (outdoor) coil | `p10_condenser_coil.png` | Dirty condenser coil |
| 11 ★ | Outdoor fan blade | `p11_fan_blade.png` | Outdoor fan failure |
| 12 ★ | Outdoor fan motor | `p12_fan_motor.png` | Outdoor fan failure |
| 13 ★ | Fan capacitor (small) | `p13_fan_capacitor.png` | Outdoor fan failure |
| 14 | Compressor | `p14_compressor.png` | (affected by Failed run capacitor) |
| 15 ★ | Compressor run capacitor (large) | `p15_run_capacitor.png` | No fault, Failed run capacitor |
| 16 ★ | Service valves and flare nuts | `p16_service_valves.png` | Low refrigerant |
| 17 ★ | Copper pipes, insulated | `p17_copper_lines.png` | Low refrigerant |
| 18 | Contactor (relay) | `p18_contactor.png` | - |

## Stage 3: fault images, `library/<fault>/photos/`

| Fault | File | Made from | Shows | `shows` step |
|---|---|---|---|---|
| No fault | `normal/photos/photo_01.png` | p02 | Clean filter held up to the light | `inspect_filter` |
| No fault | `normal/photos/photo_02.png` | p03 | Clean, dry coil | `inspect_coil` |
| No fault | `normal/photos/photo_03.png` | p15 | Healthy capacitor, flat top | `inspect_capacitor` |
| Clogged filter | `clogged_air_filter/photos/photo_01.png` | p02 | Thick grey dust mat on the filter | `inspect_filter` |
| Clogged filter | `clogged_air_filter/photos/photo_02.png` | p03 | Frost on the lower part of the coil | `inspect_coil` |
| Outdoor fan failure | `condenser_fan_failure/photos/photo_01.png` | p12 | Burnt motor winding, blackened copper | `inspect_fan_motor` |
| Outdoor fan failure | `condenser_fan_failure/photos/photo_02.png` | p11 | Rusted fan hub and shaft, blade stopped | `watch_outdoor_fan` |
| Outdoor fan failure | `condenser_fan_failure/photos/photo_03.png` | p13 | Small capacitor, bulged and leaking | `measure_fan_capacitor` |
| Low refrigerant | `low_refrigerant/photos/photo_01.png` | p16 | Oily, dust-caked flare nut | `inspect_joints` |
| Low refrigerant | `low_refrigerant/photos/photo_02.png` | p17 | Green corrosion and pinhole on a copper bend | `inspect_joints` |
| Low refrigerant | `low_refrigerant/photos/photo_03.png` | p03 | Frost only at the pipe end of the coil | `inspect_coil` |
| Dirty condenser coil | `dirty_condenser_coil/photos/photo_01.png` | p10 | Fins packed with dust, lint and leaves | `inspect_condenser_fins` |
| Dirty condenser coil | `dirty_condenser_coil/photos/photo_02.png` | p10 | Bent, flattened fins, close up | `inspect_condenser_fins` |
| Failed run capacitor | `failed_compressor_capacitor/photos/photo_01.png` | p15 | Domed top, scorch marks, oil leak | `inspect_capacitor` |
| Failed run capacitor | `failed_compressor_capacitor/photos/photo_02.png` | p15 | Meter reading well below the label | `measure_capacitor` |
| Faulty room sensor | `faulty_room_thermistor/photos/photo_01.png` | p06 | Sensor head cracked, wire insulation broken | `measure_thermistor` |
| Faulty room sensor | `faulty_room_thermistor/photos/photo_02.png` | p06 | Sensor hanging loose from its clip | `measure_thermistor` |
| Faulty room sensor | `faulty_room_thermistor/photos/photo_03.png` | p09 | Display 18°C, thermometer beside it 27°C | `compare_thermometer` |

Each fault folder also needs `photos.json`, for example:

```json
{
  "photo_01.png": {"source": "chatgpt", "shows": "inspect_filter"},
  "photo_02.png": {"source": "chatgpt", "shows": "inspect_coil"}
}
```
