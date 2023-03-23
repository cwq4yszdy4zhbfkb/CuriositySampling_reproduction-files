#!/bin/bash
#SBATCH -J MD
#SBATCH -p YOUR_PARTITION
#SBATCH --time=15000:00:00
#SBATCH --gres=gpu:1
#SBATCH -o slur.out
#SBATCH -e slurm.out
#SBATCH --ntasks 1
#SBATCH -c 4
##SBATCH --ntasks-per-node 16
##SBATCH --exclusive
#SBATCH -N 1
##SBATCH --gres-flags=enforce-binding
WORKDIR=`pwd`

source ~/.bashrc
conda activate your_openmm_env
cd $WORKDIR
python $WORKDIR/run.py


