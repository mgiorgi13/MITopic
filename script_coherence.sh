#!/bin/bash

source venv/Scripts/activate

for i in {2010..2022}
do
    python mit_topics.py g $i 8
    echo -e "year : '$i'\n"
done
echo "COMPLETE"
sleep 10