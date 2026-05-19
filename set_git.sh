#!/bin/bash

# Script to configure git user based on current user
# Usage: source set_git.sh
USER_NAME="$1"

if [[ "$USER_NAME" == "tony" ]]; then
    git config --local user.name "Tony@DROID"
    git config --local user.email "tonyw3@seas.upenn.edu"
elif [[ "$USER_NAME" == "subin" ]]; then
    git config --local user.name "Subin@DROID"
    git config --local user.email "21superbaby@gmail.com"
else
    echo "Revert to default user"
    git config --local user.name "Franka@DROID"
    git config --local user.email "frsh@seas.upenn.edu"
fi

echo "Git user set to $USER_NAME"
git config user.name && git config user.email
