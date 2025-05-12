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

regexSource="regex101"
regexConstruction="[0-9]"
regexConstructionThresholdLower="0.1"
regexConstructionThresholdUpper="0.2"

python3 similarity.py \
--regexGroup "same_construction_percentage" \
--regexConstruction "$regexConstruction" \
--regexSource "$regexSource" \
--regexConstructionThresholdLower "$regexConstructionThresholdLower" \
--regexConstructionThresholdUpper "$regexConstructionThresholdUpper"