#!/bin/bash

ORGANIZATION=$ORGANIZATION
ACCESS_TOKEN=$ACCESS_TOKEN

# REG_TOKEN=$(curl -sX POST -H "Authorization: token ${ACCESS_TOKEN}" https://api.github.com/orgs/${ORGANIZATION}/actions/runners/registration-token | jq .token --raw-output)

cd /home/docker/actions-runner

./config.sh --url https://github.com/${ORGANIZATION} --token ${ACCESS_TOKEN}

remove() {
    echo "Removing runner..."
    # ./config.sh remove --unattended --token ${ACCESS_TOKEN}
    ./config.sh remove --token ${ACCESS_TOKEN}
}

trap 'remove; exit 130' INT
trap 'remove; exit 143' TERM

./run.sh & wait $!
