# ALBERT DWAMENA-POKU — COLAB/LOCAL READY

Anchor: REM-ALBERT\_DWAMENA\_POKU-2026-08-26

## Included data

* `data/ALBERT\_DWAMENA\_POKU\_UCI\_STUDENT\_DATA.csv` — 4,424-row UCI benchmark, normalized to comma-separated CSV.
* `data/ALBERT\_DWAMENA\_POKU\_GHANA\_INSTITUTIONAL\_DATA.csv` — 320 anonymized institutional course records combined from the five supplied course files. Direct identifiers are excluded.
* `data/ALBERT\_DWAMENA\_POKU\_TEACHER\_TRUST\_DATA.csv` — supplied 350-row lecturer analytical file. Inspect `Record Status` before describing provenance.

## Run order

0 → 1 → (2, 3) → (4, 5) → 6 → 7 → 8. Scripts 2, 5, 7 and 8 refit nothing.

## Google Colab

1. Upload/extract this whole folder into Colab Files or Google Drive.
2. Open `ALBERT\_DWAMENA\_POKU\_GOOGLE\_COLAB.ipynb`.
3. Set the notebook working directory to this folder if needed.
4. Run all cells.

## Windows

Double-click `run\_albert\_windows.bat`, or run:

```
python -m pip install -r requirements.txt
python RUN\_ALBERT\_DWAMENA\_POKU\_LOCAL.py
```

## Linux/macOS

```
chmod +x run\_albert\_linux\_mac.sh
./run\_albert\_linux\_mac.sh
```

## Quick smoke test

Use `python RUN\_ALBERT\_DWAMENA\_POKU\_SMOKE\_TEST.py`. It uses 3-fold CV only for testing portability. Normal runs retain the configured 10-fold CV.

## Phase 3 gate

Script 6 intentionally reports NOT RUN until `REG\_PARAMS` and `FEATURE\_REFINEMENT\_K` are filled with methodology-approved values. It does not fabricate defaults.

## Important provenance note

The supplied lecturer file has 350 rows, but its `Record Status` column must be audited before the manuscript states all 350 were genuine collected lecturer responses.

