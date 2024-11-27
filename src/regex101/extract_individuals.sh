#!/bin/bash

# extract all fragments from each page to get individual regexes
mkdir -p regex101/regexes/
COUNTER=0
jq -cr '.data[] | (.permalinkFragment + " https://regex101.com/api/regex/" + .permalinkFragment + "/" + (.version | tostring))' regex101/pages/*.json | \
    while read -r frag url;
    do
        COUNTER=$[$COUNTER +1]
        printf "$COUNTER | "
        # If the regex has not already been fetched, fetch it
        [ -f "regex101/regexes/$frag.json" ] || (wget -nv -O "regex101/regexes/$frag.json" -nc "$url"; sleep 1)
    done