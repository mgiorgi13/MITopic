#!/bin/bash



for i in {2014..2022}
do
    python mit_topics.py bc $i 8
    echo -e "year : '$i'\n"
done
echo "COMPLETE"
sleep 10