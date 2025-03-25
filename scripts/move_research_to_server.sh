#!/bin/bash

current_path="$(pwd)"

username="$1"
server_host="$2"
server_path="$3"

[[ "$current_path" =~ scripts$ ]] && cd ..

move_via_scp () {
  if [[ "$1" == "-r" ]];
    then
      scp -O -r "$2" "$username"@"$server_host":"$server_path"/cred/"$2"/
      echo "move folder $2"
    else
      scp "$1" "$username"@"$server_host":"$server_path"/cred/"$1"
      echo "move file $1"
  fi
}

# move databases to server
scp -s ./*.db "$username"@"$server_host":"$server_path"/cred/

# move clustering research files to server
for f in research/clustering/*
do
  if [[ -d "$f" ]]; then
    [[ "$f" =~ html-to-pdf|pycache|.ipynb ]] || move_via_scp -r "$f"
  fi
  case $f in
    *.py) move_via_scp "$f";;
    *.txt) move_via_scp "$f";;
    *.yml) move_via_scp "$f";;
    *.sh) move_via_scp "$f";;
  esac
done