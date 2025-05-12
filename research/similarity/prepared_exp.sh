#!/bin/bash

python_activate() {
  . "../../.venv/bin/activate" || exit
}

clear_old_results() {
	! [ -d results ] || sudo rm -r results
}

python_activate

clear_old_results

! [ -f "../../regex101.db" ] && { echo "regex101.db is not found!"; exit; }
! [ -f "../../regexlib.db" ] && { echo "regexlib.db is not found!"; exit; }

regexSource="regexlib"
regexConstruction="[0-9]"
regexConstructionThreshold="0.3"

python3 similarity.py \
--regexGroup "same_construction_percentage" \
--regexConstruction "$regexConstruction" \
--regexSource "$regexSource" \
--regexConstructionThreshold "$regexConstructionThreshold"