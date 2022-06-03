#!/bin/bash

source venv/Scripts/activate

for i in {1990..1995}
do
    echo -e "year : '$i'\n"
    python mit_topics.py bc $i 8
done
echo "COMPLETE"
sleep 10