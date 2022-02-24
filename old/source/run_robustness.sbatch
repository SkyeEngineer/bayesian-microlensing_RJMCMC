#!/bin/bash -e

#SBATCH --mail-user=jyar540@aucklanduni.ac.nz
#SBATCH --mail-type=ALL
#SBATCH --job-name=flow_microlensing
#SBATCH --account=uoa03400
#SBATCH --output=robustness.out
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2

module load Miniconda3/4.10.3 

source ~/.bashrc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scale_wlg_nobackup/filesets/nobackup/uoa03400/microlensing/pytorch_env/lib/

conda activate ./pytorch_env

python robustness.py