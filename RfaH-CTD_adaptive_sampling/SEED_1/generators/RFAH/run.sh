#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate YOUROPENMMENV
python openmm_run.py
