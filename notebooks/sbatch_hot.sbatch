#!/bin/bash

#SBATCH --job-name=Array1d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=08:00:00
#SBATCH --array=0-985

if [ "$SLURM_JOBTMP" != "" ]; then
    THEANO_FLAGS="base_compiledir=$SLURM_JOBTMP,$THEANO_FLAGS"
fi

export THEANO_FLAGS="$THEANO_FLAGS"


if [ "$SLURM_ARRAY_TASK_ID" == "" ]; then exit; fi

i=0
while read -r line; do
    if [ $i -eq $SLURM_ARRAY_TASK_ID ]; then

		module purge
		module load anaconda3/5.3.1
		source ~/pyenv/py3.7.0/bin/activate

        eval $line 
        exit
    fi
    i=$((i+1))
done < sbatch_hot.txt

exit
