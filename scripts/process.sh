#!/bin/bash
#
#SBATCH -p test # partition (queue)
#SBATCH -c 8 # number of cores
#SBATCH --mem 100g # memory pool for all cores
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH -o scripts/slurm/PTBPE_scan.%j.out # STDOUT
#SBATCH -e scripts/slurm/PTBPE_scan.%j.err # STDERR


shopt -s globstar

# Directory containing the files
DIR=../data/${1}

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    echo "Directory $DIR not found. Exiting."
    exit 1
fi

count=0
max=100

# Loop over every file in the directory
for file in "$DIR"/**/*.pdb; do
    if [ -f "$file" ]; then
        ((count++))
        if [ "$count" -gt "$max" ]; then
            echo "Reached $max files, stopping."
            break
        fi

        stem="${file%.*}"
        echo "Processing file #$count: $file (stem: $stem)"
        ./scan.sh "$stem"
    fi
done
