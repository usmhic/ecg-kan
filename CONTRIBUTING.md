# Contributing Guidelines

## Branching and Commits

- Use short feature branches.
- Keep commits focused and atomic.
- Write commit messages in imperative mood (for example: `Add WTMM benchmark export`).

## Reproducibility Requirements

Before opening a PR:

1. Ensure environment setup works from `environment.yml`.
2. Re-run the relevant notebook/script section.
3. Verify benchmark artifacts are consistent with manuscript claims.
4. Remove local cache artifacts before committing.

## Notebook Hygiene

- Keep notebook cell order executable from top to bottom.
- Avoid committing unnecessary large outputs.
- Use clear section headers and parameter cells.

## Data Handling

- Do not commit raw data dumps or redundant dataset copies.
- Use PhysioNet as the source of truth for ECG-ID.
- Keep only lightweight, publication-relevant derived files.

## Documentation Updates

When changing pipeline logic, update:

- `README.md`
- `docs/REPOSITORY_STRUCTURE.md`
- `docs/RESEARCH_REFERENCE.md`
- manuscript text in `paper/main.tex` when results interpretation changes
