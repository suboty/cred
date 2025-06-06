#!/bin/bash

python_activate() {
  . "../../.venv/bin/activate" || exit
}

clear_old_results() {
	! [ -d tmp ] || sudo rm -r tmp
}

python_activate

clear_old_results

! [ -f "../../regexlib.db" ] && { echo "regexlib.db is not found!"; exit; }

regexSource="regexlib"
regexConstruction="[0-9]"
regexConstructionThresholdLower="0.3"
regexConstructionThresholdUpper="0.5"

python3 similarity.py \
--regexGroup "same_construction_percentage" \
--regexConstruction "$regexConstruction" \
--regexSource "$regexSource" \
--regexConstructionThresholdLower "$regexConstructionThresholdLower" \
--regexConstructionThresholdUpper "$regexConstructionThresholdUpper"