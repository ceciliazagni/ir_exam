#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=EPYC
#SBATCH --mem=400G
#SBATCH --time=10:00:00
#

module load conda
conda activate irpy3

python3 interactive_update_the_dictionary_cz.py
