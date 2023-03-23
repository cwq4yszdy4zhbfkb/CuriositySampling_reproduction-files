#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate YOUR_OPENMM_ENV
python openmm_run.py
