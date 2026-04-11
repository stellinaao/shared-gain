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

If the hooks modified a file, just make sure to git add and commit again.

## File Annotations
Please add the following to the head of scripts and notebooks.

```
"""
filename.py

Description of script.

Author: Fname Lname
Created: yyyy-mm-dd
Last Modified: yyyy-mm-dd
Python Version: 3.11.14
"""
```

e.g.,

```
sandbox.ipynb

A sandbox to play around with new analyses.

Author: Stellina X. Ao
Created: 2026-03-05
Last Modified: 2026-03-23
Python Version: 3.11.14
```
