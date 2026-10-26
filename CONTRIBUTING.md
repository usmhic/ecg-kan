# Contributing

Keep changes focused and explain what changes in the method or public interface.
Use short branches and clear commit messages.

## Validate a change

```text
python -m unittest discover -s tests -v
python research/build_paper_tables.py --check
python research/audit_artifacts.py
```

For pipeline changes, also run a real-data smoke test in an ignored output
folder. Run affected notebook cells from top to bottom. For manuscript changes,
compile `paper/main.tex` twice with `pdflatex -halt-on-error` from `paper/`.

Keep notebooks free of large outputs and private local paths. Shared method
logic belongs in the Python modules; notebooks should call those functions.

## Preserve the research record

The committed CSVs are historical results. Do not overwrite them during a smoke
test, or change values merely to make a check pass. New experiments should use a
separate `--out-dir`. Changing models, preprocessing, thresholds, sampling or
seeds requires a complete relevant rerun before updating numerical claims.
Generate manuscript tables with `python research/build_paper_tables.py` after
reviewing new metrics. Keep the paper, results guide and README consistent.

Dependency ranges are not a historical lockfile. New run configurations capture
the installed environment and source hashes. Record failures and warnings.

## Data and attribution

Download ECG-ID from PhysioNet under its own licence and cite its creator.
Do not commit the raw database, environments, logs, credentials or local caches.
Retain only permitted, useful derived artifacts. Source author contributions,
funding and ethics declarations from the authors rather than assumptions.
