#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH -p YOURPARTITION
#SBATCH --gres=gpu:4
#SBATCH --time=2000:00:00
##SBATCH --mem=50GB
#SBATCH --exclusive
#SBATCH --output=jupyter.log
x=`pwd`
source ~/.bashrc
conda activate YOUR_CURIOSITY_ENV
XDG_RUNTIME_DIR=""
cd $x
jupyter lab --no-browser --port=3333
wait 
