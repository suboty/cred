#!/bin/bash

# download each page of search results
mkdir -p regex101/pages/
# get first regex (for metadata)
wget "https://regex101.com/api/library/1/?orderBy=MOST_POINTS&search=" -nv -O regex101/pages/1.json
# parse json for getting number of pages
PAGES=$(jq -r .pages regex101/pages/1.json)
echo "--- find $PAGES pages\n"
for i in $(seq 2 $PAGES); do
    # fetch this page of regular expressions
    printf "$i / $PAGES | "
    wget "https://regex101.com/api/library/$i/?orderBy=MOST_POINTS&search=" -nv -O "regex101/pages/$i.json"
    sleep 1
done