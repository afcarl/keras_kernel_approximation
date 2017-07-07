#!/usr/bin/env bash

export PYTHONPATH=`pwd`:${PYTHONPATH}
python prepare_folders.py
python xp/train_rff.py
