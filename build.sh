#!/bin/bash

venv="feet-virtualenv"

echo "building virtualenv: $venv"

hash virtualenv
if [ "$?" != "0" ];
  then
    pip install virtualenv;
fi

virtualenv $venv

echo "installing feet"
$venv/bin/pip install -e .

