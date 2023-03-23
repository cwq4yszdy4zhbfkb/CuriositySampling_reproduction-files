#!/bin/bash
#SBATCH -J WLALL5
#SBATCH -p YOUR_PARTITION
#SBATCH --time=15000:00:00
#SBATCH --gres=gpu:1
#SBATCH -o OUT_INFO.out
#SBATCH -e OUT_ERROR.out
#SBATCH --mem 25000
##SBATCH --ntasks 4
##SBATCH -c 4
##SBATCH --ntasks-per-node 16
##SBATCH --exclusive
#SBATCH -N 1
##SBATCH -x cnm01
##SBATCH --gres-flags=enforce-binding
WORKDIR=`pwd`

source ~/.bashrc
conda activate YOUR_ENV_WITH_CURIOSITY
cd $WORKDIR

python $WORKDIR/curiositcalc_rep20.py



