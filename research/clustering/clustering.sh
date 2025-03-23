#!/bin/bash

if [ "$#" -eq 0 ]; then
  set -- "_"
  echo "Working with the whole dataset"
fi

python_activate() {
  . "../../.venv/bin/activate" || exit
}

clear_old_reports() {
	! [ -d tmp ] || sudo rm -r tmp
}

python_activate

clear_old_reports

str="="
lineStr=$(printf "$str%.0s" {1..30})

for arg in "$@"; do
  if [ "$arg" != "_" ]; then
    echo "Experiment with <$arg> filter word"
  fi
	python3 clustering.py -v -u -e -n --algname "tf_idf|bert" --filter "$arg"
	echo "$lineStr"
done

sh html_to_pdf.sh
