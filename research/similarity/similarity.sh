#!/bin/bash

if [ "$#" -eq 0 ]; then
  set -- "all"
fi

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

read -r -p "Enter regex source: " regexSource

for arg in "$@"; do
  echo "Experiment with <$arg> regex group"
  if [ "$arg" != "same_construction" ];
  then
    python3 similarity.py \
    --regexGroup "$arg" \
    --regexSource "$regexSource"
  else
    read -r -p "Enter regex construction: " regexConstruction
    read -r -p "Do you want to filter by percentage? (y/n): " percentageChoice
    if [ "$percentageChoice" = "yes" ]; then
      percentageChoice="y"
    fi
    if [ "$percentageChoice" = "y" ]; then
      read -r -p "Enter percentage threshold: " regexConstructionThreshold
      python3 similarity.py \
      --regexGroup "same_construction_percentage" \
      --regexConstruction "$regexConstruction" \
      --regexSource "$regexSource" \
      --regexConstructionThreshold "$regexConstructionThreshold"
    else
      python3 similarity.py \
      --regexGroup "$arg" \
      --regexConstruction "$regexConstruction" \
      --regexSource "$regexSource"
    fi
  fi
	lineStr
done
