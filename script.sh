#!/bin/bash

source venv/Scripts/activate

for i in {2002..2013}
do
    python mit_topics.py bc $i 8
    echo -e "year : '$i'\n"
done
echo "COMPLETE"
sleep 10