#!/bin/bash
#SBATCH --job-name=aPENT
#SBATCH -p YOUR_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --time=2000:00:00
##SBATCH --mem=50GB
##SBATCH --exclusive
#SBATCH --output=jupyter.log
x=`pwd`
source ~/.bashrc
conda activate htmd_env
cd $x
python Run_AdaptiveSampling.py
wait 
