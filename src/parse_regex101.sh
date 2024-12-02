#!/bin/bash

# parse all pages MOST_POINTS
sh src/regex101/parse_pages.sh

# extract individuals from pages
sh src/regex101/extract_individuals.sh

# migrate regexes to db
python3 src/regex101/migrate_regexes.py