#!/bin/bash

if [ "$#" -eq 0 ]; then
  set -- "_"
fi

python_activate() {
  . "../../.venv/bin/activate" || exit
}

clear_old_reports() {
	! [ -d tmp ] || sudo rm -r tmp
}

python_activate

clear_old_reports

! [ -f "../../regex101.db" ] && { echo "regex101.db is not found!"; exit; }

lineStr () {
  for _ in $(seq 1 30); do
    printf "="
  done
  printf "\n"
}

read -r -p "Enter regex source: " regexSource
read -r -p "Enter clustering algorithm: " clusteringAlg

read -r -p "Is need to save regular expressions? (y/n): " isRegexesSaving
read -r -p "Is need to save clustering reports? (y/n): " isClusteringReportsSaving

if [ "$isRegexesSaving" = "yes" ]; then
  isRegexesSaving="y"
fi

if [ "$isClusteringReportsSaving" = "yes" ]; then
  isClusteringReportsSaving="y"
fi

for arg in "$@"; do
  if [ "$arg" != "_" ];
  then
    echo "Experiment with <$arg> filter word"
    python3 clustering.py -s -e -n \
    --algName "tf_idf|bert" \
    --clusteringName "$clusteringAlg" \
    --filter "$arg" \
    --clustersNum 50 \
    --regexSource "$regexSource" \
    --isClusteringReportsSaving "$isClusteringReportsSaving" \
    --isRegexesSaving "$isRegexesSaving"
  else
    echo "Working with the whole dataset"
    python3 clustering.py -s -e -n \
    --algName "tf_idf|bert" \
    --clusteringName "$clusteringAlg" \
    --filter "$arg" \
    --clustersNum 50 \
    --clusterStep 5 \
    --clusterStart 5 \
    --regexSource "$regexSource" \
    --isClusteringReportsSaving "$isClusteringReportsSaving" \
    --isRegexesSaving "$isRegexesSaving"
  fi
	lineStr
done

if [ "$isClusteringReportsSaving" = "y" ]; then
  sh html_to_pdf.sh
fi
