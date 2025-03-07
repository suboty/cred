#!/bin/bash

python_activate() {
  . "../../.venv/bin/activate" || exit
}

clear_old_reports() {
	sudo rm -r assets || sudo rm -r clustering_reports
}

python_activate

clear_old_reports

str="="
lineStr=$(printf "$str%.0s" {1..30})

for arg in "$@"; do
    echo "Experiment with <$arg> filter word"
	python3 clustering.py -v -u --algname tf_idf --filter $arg
	python3 clustering.py -v -u --algname bert --filter $arg
	echo "$lineStr"
done
