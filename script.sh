#!/bin/bash

source venv/Scripts/activate

for i in {1990..2023}
do
    python mit_topics.py bc $i 8
    echo -e "year : '$i'\n"
done
echo "COMPLETE"
sleep 10