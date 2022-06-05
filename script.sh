#!/bin/bash

source venv/Scripts/activate

for i in {1990..2022}
do
    echo -e "year : '$i'\n"
    python mit_topicGlobal.py lda $i 8
done
echo "COMPLETE"
sleep 10