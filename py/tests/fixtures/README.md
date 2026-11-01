# Test fixtures

`crm-hl.meshinfo.json` and `heat-exchanger.meshinfo.json` are **synthetic**: they describe no real mesh. Their counts (48.2 M and 1.2 M cells), zones, bounds and quality are made up to exercise the SDK, and their `source.sha256` is the hash of the stand-in bytes `tests/conftest.py` writes as the mesh. They are not the HLPW-5 CRM-HL grid; that grid, its MeshInfo and its zone groups are V&V's (`vv/`).
