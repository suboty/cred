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

for arg in "$@"; do
  if [ "$arg" != "_" ];
  then
    echo "Experiment with <$arg> filter word"
    python3 clustering.py -s -e -n \
    --algname "tf_idf|bert" \
    --clusteringname "$clusteringAlg" \
    --filter "$arg" \
    --clustersnum 50 \
    --source "$regexSource"
  else
    echo "Working with the whole dataset"
    python3 clustering.py -s -e -n \
    --algname "tf_idf|bert" \
    --clusteringname "$clusteringAlg" \
    --filter "$arg" \
    --clustersnum 50 \
    --clusterstep 5 \
    --clusterstart 5 \
    --source "$regexSource"
  fi
	lineStr
done

sh html_to_pdf.sh
