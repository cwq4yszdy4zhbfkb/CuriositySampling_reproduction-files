#!/bin/bash
#SBATCH --job-name=bPENT
#SBATCH -p YOURPARTITION
#SBATCH --gres=gpu:1
#SBATCH --time=2000:00:00
##SBATCH --mem=50GB
##SBATCH --exclusive
#SBATCH --output=jupyter.log
x=`pwd`
source ~/.bashrc
conda activate htmdenv
cd $x
python Run_AdaptiveBandit.py
wait 
