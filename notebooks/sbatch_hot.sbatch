#!/bin/bash

#SBATCH --job-name=Array1d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=01:00:00
#SBATCH --array=1-315

if [ "$SLURM_ARRAY_TASK_ID" == "" ]; then exit; fi

module purge
module load anaconda3/5.3.1

# pip install astropy fitsio lightkurve --user # should be ok for my own login but uncomment if you need to install
# pip install https://github.com/dfm/acor/archive/master.zip https://github.com/OxES/OxKeplerSC/archive/master.zip --user

i=0
while read -r line; do
    i=$((i+1))
    if [ $i -eq $SLURM_ARRAY_TASK_ID ]; then
	eval "$line"
    fi
done < t1.txt
