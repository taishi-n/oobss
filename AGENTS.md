# Repository Guidelines

## Project Structure & Module Organization
- Source code: `src/oobss/` with namespaces:
  - `oobss.separators` (algorithms), `oobss.audio` (I/O, STFT), `oobss.configs`, `oobss.logging_utils`.
- CLI entrypoint: `oobss` (defined in `pyproject.toml`).
- Documentation: `docs/` (Material for MkDocs + mkdocstrings).
- Examples: `examples/` (separation, evaluation, benchmarks, block spectrogram).
- Tests: `tests/`.

## Build, Test, and Development Commands
- `uv sync` — create/update the virtual environment.
- `uv run pytest` — run the test suite.
- `uv run mkdocs build` — build docs locally.
- `uv run ruff format .` — auto-format code.
- `uv run ty check` — type check.
- Releases: `uv build` / `uv publish` (must still support `pip install torchrir`).
- `uv.lock` is committed; update it **only** when `pyproject.toml` changes.

## Coding Style & Naming Conventions
- Python: follow Ruff defaults and `ty` type checker; prefer explicit imports and small modules.
- Naming: snake_case for functions/variables, PascalCase for classes. Public API lives under the `oobss` package.
- Keep docstrings concise; use English for all docs and comments.

## Testing Guidelines
- Framework: `pytest`.
- Run: `uv run pytest`.
- Aim to cover separator interfaces (`OnlineSeparator.process_stream`), streaming utilities, and examples where feasible.
- Keep fixtures small; prefer synthetic data over large assets.

## Commit & Pull Request Guidelines
- Codex proposes commit message drafts; the user reviews/approves before committing.
- All commits must follow Conventional Commits via Commitizen (`uv run cz commit`).
- Before PR: run format (`uv run ruff format .`), type check (`uv run ty check`), tests, and (if docs touched) `uv run mkdocs build`.
- PRs should include a concise summary, linked issues/design notes, and example outputs/benchmarks when performance-critical code changes.

## Versioning & Releases
- Use Commitizen for version bumps, tagging, and changelog updates.
  - Run: `uv run cz bump --changelog --yes`
  - Version source: `pyproject.toml` (PEP 621); tag format `vX.Y.Z`.
- After bumping, push commits and tags: `git push origin main --tags`.
- Signed tags are manual by the user, e.g., `git tag -s vX.Y.Z -m "vX.Y.Z"`.

## Security & Configuration Tips
- Secrets/credentials should never be committed; use environment variables for local tests.
- Optional deps (torch, torchaudio, torchrir, sounddevice) are not pinned; document which extras are required for your feature.
