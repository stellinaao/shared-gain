# Contributing Guide
Last modified: 04/05/2026 (Stellina Ao)


This repo contains shared infrastructure and many individual analyses. Please abide by the following guidelines to keep collaboration smooth and painless.

## Repo Structure
```
src/
--core/             # shared code (e.g., data loading, preprocessing, gen-purpose utils)
--<analysis_name>   # analysis-specific code (e.g., stellina's shared gain, letizia's communication subspace, etc.)

notebooks/
--<analysis_name>   # contains notebooks specific to an analysis (e.g., stellina's sandbox.ipynb for exploratory analyses)

scripts/
--<analysis_name>   # contains scripts specific to an analysis
```

## Shared v. Private Code
`src/core` contains reusable code, while `src/<analysis_name>`, `notebooks/<analysis_name>`, and `scripts/<analysis_name>` contain code specific to individual downstream analyses. Please do not modify others' analyses folders without discussion. If you write a general-purpose function or script that might be useful to others, please add it to `src/code` with a pull request.

## Pre-Commit Hooks
To keep code clean and consistent across collaborators, please utilize the pre-commit hooks. Run the following in the CLI before you commit for the first time. Afterwards, the hooks should run everytime you commit.
```
pre-commit install
pre-commit run --all-files
```
