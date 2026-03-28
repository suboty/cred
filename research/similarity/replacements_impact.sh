#!/bin/bash

python_activate() {
  . "../../.venv/bin/activate" || exit
}

python_activate

! [ -f "../../regex101.db" ] && { echo "regex101.db is not found!"; exit; }

lineStr () {
  for _ in $(seq 1 30); do
    printf "="
  done
  printf "\n"
}

python3 replacements_impact.py
lineStr
