Static Eurovision community map export
=====================================

This folder contains a standalone exporter for a static geographic community map.

Default output:

- `community_world_map_1975_2025_min21.png`
- `community_world_map_1975_2025_min21.csv`

Run it with:

```bash
python static_exports/generate_community_world_map.py
```

Optional HTML fallback:

```bash
python static_exports/generate_community_world_map.py --also-html
```

The exporter uses the same year range and participation threshold as the app by default:

- Years: 1975 to 2025
- Minimum participation: 21 years