#!/bin/bash

project="feet"

venv="$project-virtualenv"

echo "building virtualenv: $venv"

hash virtualenv
if [ "$?" != "0" ];
  then
    pip install virtualenv;
fi

virtualenv $venv
source $venv/bin/activate

pip install -e .

