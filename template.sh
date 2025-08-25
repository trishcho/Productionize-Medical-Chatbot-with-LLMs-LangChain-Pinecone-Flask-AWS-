#!/usr/bin/env bash
set -euo pipefail

# Creating directory
mkdir -p src research

# Creating files
: > src/__init__.py
: > src/helper.py
: > src/prompt.py
: > .env
: > setup.py
: > app.py
: > research/trials.ipynb
: > requirements.txt

echo "Directory and files created successfully!"
