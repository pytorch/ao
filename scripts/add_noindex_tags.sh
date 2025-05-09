#!/bin/bash

# Adds <meta name="robots" content="noindex"> tags to all html files in a
# directory (recursively)
#
# Usage:
# ./add_noindex_tags.sh <directory>
#
# Example (from the root directory if previous release was 0.3)
# ./scripts/add_noindex_tags.sh 0.3
if [ "$1" == "" ]; then
  echo "Incorrect usage. Correct Usage: add_no_index_tags.sh <directory>"
  exit 1
fi
find $1 -name "*.html" -print0 | xargs -0 sed -i '/<head>/a \ \ <meta name="robots" content="noindex">'
