# Photos for the job simulator

Put photos in `library/<condition>/photos/` and describe them in that folder's `photos.json`.
A photo appears in the game when the technician performs the step named in `shows`.

```json
{
  "photo_01.png": {"source": "chatgpt", "shows": "inspect_filter"},
  "photo_02.png": {"source": "muse-image", "shows": "inspect_coil"}
}
```

`source` is one of `real-photo`, `chatgpt` or `muse-image`. The app shows it on every photo.

## Which photo for which fault

| Folder | Photo | `shows` |
|---|---|---|
| `normal` | Clean filters held up to the light | `inspect_filter` |
| `normal` | Clean, dry indoor coil | `inspect_coil` |
| `clogged_air_filter` | Filter covered in grey dust | `inspect_filter` |
| `clogged_air_filter` | Frost on the lower part of the indoor coil | `inspect_coil` |
| `condenser_fan_failure` | Outdoor fan blade standing still behind the grille | `watch_outdoor_fan` |
| `condenser_fan_failure` | Fan capacitor close-up (or seized fan motor) | `inspect_capacitor` |
| `low_refrigerant` | Oily, dust-caked residue on a flare nut / service valve | `inspect_joints` |
| `low_refrigerant` | Frost only on the inlet end of the indoor coil | `inspect_coil` |
| `dirty_condenser_coil` | Outdoor coil fins packed with dust, lint and leaves | `inspect_condenser_fins` |
| `failed_compressor_capacitor` | Run capacitor with a domed top and scorch marks | `inspect_capacitor` |
| `failed_compressor_capacitor` | Capacitance meter showing a low reading on the capacitor | `measure_capacitor` |
| `faulty_room_thermistor` | Room sensor hanging loose from its clip near the coil | `measure_thermistor` |
| `faulty_room_thermistor` | Remote/display showing a much lower room temperature than a thermometer beside it | `compare_thermometer` |

## All step names

`listen_outdoor`, `watch_outdoor_fan`, `listen_compressor`, `feel_supply_air`, `thermal_indoor`,
`thermal_outdoor`, `read_sensors`, `compare_thermometer`, `gauge_pressures`, `inspect_filter`,
`inspect_coil`, `inspect_condenser_fins`, `inspect_joints`, `inspect_capacitor`, `spin_fan_blade`,
`measure_capacitor`, `measure_fan_capacitor`, `measure_thermistor`

The exploded-view diagram goes in `library/diagram/split_ac_exploded.png`.
