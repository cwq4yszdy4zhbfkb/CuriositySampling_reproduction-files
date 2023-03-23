#!/bin/bash
#SBATCH -J "100nsChig"
#SBATCH -p YOURPARTITION
#SBATCH --time=15000:00:00
#SBATCH --gres=gpu:4
#SBATCH -o slur.out
#SBATCH -e slurm.out
##SBATCH --ntasks 4
##SBATCH -c 4
##SBATCH --ntasks-per-node 16
#SBATCH --exclusive
#SBATCH -N 1
##SBATCH --gres-flags=enforce-binding
WORKDIR=`pwd`

source ~/.bashrc
conda activate your_curiosity_env
cd $WORKDIR

python $WORKDIR/curiositcalc_rep20.py


