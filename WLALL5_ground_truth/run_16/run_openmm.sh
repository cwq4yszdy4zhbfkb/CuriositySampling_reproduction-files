#!/bin/bash
#SBATCH -J openmm
#SBATCH -p YOURPARTITION
#SBATCH --time=15000:00:00
#SBATCH --gres=gpu:1
#SBATCH -o slur.out
#SBATCH -e slurm.out
#SBATCH --ntasks 4
#SBATCH -c 2
##SBATCH --ntasks-per-node 16
##SBATCH --exclusive
#SBATCH -N 1
##SBATCH --gres-flags=enforce-binding
WORKDIR=`pwd`

source ~/.bashrc
cd $WORKDIR
conda activate YOUR_OPENMM_ENV 

python $WORKDIR/openmm_run.py


