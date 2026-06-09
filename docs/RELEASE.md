# ComputeCop Release Checklist

This document is the release engineering guide for ComputeCop maintainers. It
describes how to cut a new version, the verification gates every release must
pass, and the supported platforms a release targets.

## Supported Platforms

Each release is verified against:

- Windows 10/11
- macOS
- Python 3.11 or newer

Linux is supported on a best-effort basis: ComputeCop runs there, but the
telemetry and thermal layers are validated primarily on Windows and macOS.
Temperature sensors are optional everywhere because sensor availability varies
by operating system and hardware; ComputeCop falls back to CPU pressure
heuristics when sensor data is unavailable.

Run `computecop doctor` on a target machine to confirm the interpreter,
platform, RAM baseline, `psutil` access, endpoint reachability, event log path,
and configuration are all healthy before release validation.

## Versioning

ComputeCop follows semantic versioning (`MAJOR.MINOR.PATCH`). The version is
declared in exactly two tracked locations, which must always agree:

- `pyproject.toml` — the `project.version` field
- `src/computecop/__init__.py` — the `__version__` constant

The README version badge (`README.md`) is documentation and is updated alongside
the version bump so published docs match the release.

## Version Bump Process

1. Confirm `main` is clean and up to date with `origin/main`:

   ```bash
   git status --short --branch
   ```

2. Update the version in both tracked locations and the README badge:

   - `pyproject.toml`
   - `src/computecop/__init__.py`
   - `README.md` version badge

3. Verify the package reports the new version:

   ```bash
   python -c "import computecop; print(computecop.__version__)"
   ```

4. Commit the bump on its own with a conventional message, for example
   `chore(release): bump version to X.Y.Z`.

5. Implement the release scope in focused, independently verifiable commits.
   Add or update tests for every behavioral change.

## Verification Gates

Every release must pass the full verification suite before it is tagged or
pushed:

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy src/computecop
python -m pytest
python -m build
```

On Windows, the same gates are available through the helper script:

```powershell
.\scripts\verify.ps1
```

A release is only complete when all five gates pass cleanly: formatting, lint,
typing, tests, and a successful source and wheel build.

## Local-Only Artifact Policy

ComputeCop keeps release planning and operational tracking files local. Public
documentation, source, and tests must never reference private planning material.
Before every commit, run `git status --short` and stage only the files that
belong to the change — never blanket-stage the working tree.

Files that are intentionally untracked or ignored, and must not be published:

- Local planning and roadmap notes
- Per-session operational trackers
- Release scratch and completion notes
- Local environment files and runtime state (for example the event JSONL log)

The `.gitignore` enumerates the ignored runtime and planning artifacts; keep it
authoritative so a stray `git add` cannot leak local-only files. Published
documentation must not mention specific personal hardware or local-only files.

## Post-Release Audit

After pushing a release, confirm:

1. Local `HEAD` matches `origin/main`.
2. The intended version appears in `pyproject.toml` and
   `src/computecop/__init__.py`.
3. No local-only planning or tracker files were staged or pushed.
4. The verification gates all passed on the released commit.
