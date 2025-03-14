#!/bin/bash

python_activate() {
  . "../../.venv/bin/activate" || exit
}

clear_old_reports() {
	sudo rm -r tmp
}

python_activate

clear_old_reports

str="="
lineStr=$(printf "$str%.0s" {1..30})

for arg in "$@"; do
    echo "Experiment with <$arg> filter word"
	python3 clustering.py -u -e -n --algname tf_idf --filter "$arg"
	python3 clustering.py -u -e -n --algname bert --filter "$arg"
	echo "$lineStr"
done
