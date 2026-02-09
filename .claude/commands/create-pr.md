---
description: Pre-review code changes, then create a well-structured PR with conventional commits
argument-hint: [branch-name]
---

# Create PR — LOATO-Bench

## Overview

Create a well-structured pull request for the LOATO-Bench project. **This command acts as a first layer of review before creating the PR** — catching issues before they reach GitHub.

## Steps If a PR Is NOT Open Yet

### 1. Understand the Changes

- Run `git status` and `git diff main...HEAD --stat` to see all changed files
- Run `git log main..HEAD --oneline` to see all commits on this branch
- If no branch exists yet (on `main`), ask the user what to name the branch

### 2. Pre-PR Quality Gate (Two Layers)

**Layer 1 — pre-commit hooks (local, runs on every commit):**

Pre-commit hooks are installed and run automatically on `git commit`. If they haven't been run yet for staged files, run them manually:

```bash
uv run pre-commit run --all-files
```

This runs: trailing whitespace, end-of-file fixer, YAML/TOML validation, large file check, merge conflict check, debug statements, ruff lint (with auto-fix), ruff format, mypy, and detect-secrets.

**Layer 2 — full QA pipeline (mirrors CI):**

After pre-commit passes, run the full test suite that CI enforces:

```bash
# Tests (must pass — CI job: test)
uv run pytest tests/ -v --tb=short
```

Note: mypy, ruff check, and ruff format are already covered by pre-commit hooks above, but CI runs them independently too. Pre-commit catches issues *before* the commit; CI catches anything that slips through.

### 3. Pre-PR Code Review

Based on which directories changed, check compliance:

**If `src/loato_bench/data/` changed:**
- UnifiedSample schema not broken (`data/base.py`)
- `configs/data/taxonomy.yaml` still valid if taxonomy.py changed
- Path traversal prevention in any file I/O

**If `src/loato_bench/analysis/` changed:**
- `docs/eda.md` kept in sync with code changes
- `configs/analysis/eda.yaml` parameters match code
- Matplotlib figures use `managed_figure()` context manager (no memory leaks)
- No raw prompt text logged (privacy)

**If `src/loato_bench/embeddings/` changed:**
- EmbeddingModel ABC interface not broken (`embeddings/base.py`)
- Cache format compatible (`embeddings/cache.py`)
- Model configs in `configs/embeddings/` updated if needed

**If `src/loato_bench/classifiers/` changed:**
- Classifier ABC interface not broken (`classifiers/base.py`)
- sklearn pipeline pattern: `StandardScaler -> classifier`
- Configs in `configs/classifiers/` updated if needed

**If `tests/` changed:**
- Tests actually test meaningful behavior (not just smoke tests)
- Test fixtures use `tmp_path` for file I/O (no leftover artifacts)
- Coverage maintained at 90%+ for non-exempt modules

**General checks (all changes):**
- No `eval()`, `exec()`, or `__import__()` on user data
- No hardcoded API keys or secrets (use `os.getenv()`)
- Type hints on all new functions
- Google-style docstrings on public functions
- No unused imports
- Line length <= 100 (ruff enforced)

### 4. Report Findings (DO NOT AUTO-FIX)

**If issues are found:**
```
I found the following issues:

**QA Failures:**
- mypy: `src/loato_bench/analysis/eda.py:42` — Incompatible return type
- ruff: `tests/test_quality.py:15` — Unused import `Any`

**Code Review:**
- `src/loato_bench/data/taxonomy.py:88` — Missing path validation on user input
- `src/loato_bench/analysis/visualization.py:120` — Figure not using managed_figure()

What would you like to do?
1. Fix these issues before creating the PR
2. Create PR anyway with the current code
```

Wait for user decision. Never auto-fix.

**If no issues found:**
Proceed to create PR.

### 5. Prepare Branch and Commit

- If changes are uncommitted, stage and commit using **conventional commits**:
  - `feat(scope):` — New feature
  - `fix(scope):` — Bug fix
  - `refactor(scope):` — Code restructuring
  - `test(scope):` — Adding/updating tests
  - `docs(scope):` — Documentation only
  - `chore(scope):` — Build, CI, config changes
  - Scopes: `data`, `eda`, `embeddings`, `classifiers`, `evaluation`, `analysis`, `cli`, `tracking`, `infra`
- If not on a feature branch, create one:
  - Pattern: `feat/<sprint>-<short-description>` (e.g., `feat/sprint-2a-taxonomy-finalization`)
  - Or: `fix/<short-description>`, `refactor/<short-description>`
- Push to remote with `-u` flag

### 6. Create PR

Use `gh pr create` with this structure:

```bash
gh pr create --title "<conventional-commit-style title>" --body "$(cat <<'EOF'
## Summary
<2-4 bullet points: what changed and why>

## Changes
<Group by area — list key files/modules affected>

### <Area 1> (e.g., Analysis, Data Pipeline, Embeddings)
- `path/to/file.py` — What changed
- `path/to/other.py` — What changed

### Tests
- `tests/test_foo.py` — X tests added/modified

## QA Status
- [x] pre-commit hooks pass (ruff, mypy, detect-secrets, file hygiene)
- [x] All tests pass (N tests)
- [ ] Manual verification needed: <describe what>

## Sprint Context
**Sprint**: <sprint number and name>
**Depends on**: <any prerequisite PRs or issues>
**Blocks**: <what can't proceed without this>

## Test Plan
- [ ] <specific things to verify>
- [ ] <edge cases to check>

---
Generated with [Claude Code](https://claude.com/claude-code)
EOF
)" --base main
```

Return the PR URL to the user.

---

## Steps If a PR IS Already Open

1. Run the same QA checks (Step 2-3 above)
2. Report any issues found (Step 4)
3. After user approval, commit new changes with conventional commit
4. Push to the existing branch (PR auto-updates)
5. Update the PR description if scope changed significantly:
   ```bash
   gh pr edit <number> --body "$(cat <<'EOF'
   <updated description>
   EOF
   )"
   ```

---

## Important Rules

- **Never auto-fix issues** — always report and ask first
- **User decides** whether to fix or proceed
- **Conventional commits only** — no freeform messages
- **Two QA layers** — pre-commit hooks (local, on every commit) + CI pipeline (GitHub Actions)
- **Pre-commit is mandatory** — if hooks aren't installed, run `uv run pre-commit install` first
- **Never skip hooks** — do not use `--no-verify` on commits
- **Privacy**: Never include raw prompt text in PR descriptions
- If `gh` auth fails, check `gh auth status` and suggest `gh auth switch`
