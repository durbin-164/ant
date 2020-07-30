#!/bin/bash
echo "starting generating coverage xml file"
if [ ! -d "coverage" ] 
then
    mkdir coverage
fi

gcovr --sonarqube

echo "Complete generating coverage xml file."