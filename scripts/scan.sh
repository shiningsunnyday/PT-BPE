#!/usr/bin/env bash

DIR=$2
# 1. Extract sequence from a local PDB (CATH domain file) using Biopython
#    We'll produce a 'sequence.fasta' file.
# python fasta.py --pdb_file ${DIR}/$1.pdb --out_file ${DIR}/${1}.fasta
FILE=$1
REL_PATH="${FILE#$DIR/}"  # strip the DIR prefix
NEW_PATH="results/$REL_PATH".pdb
if [ -f "$NEW_PATH" ]; then
    exit
fi

python fasta.py --pdb_file $1.pdb --out_file ${1}.fasta

# 2. Run InterProScan on the extracted sequence.
#    Make sure you have InterProScan installed and properly set up.
# my_interproscan/interproscan-5.73-104.0/interproscan.sh \
#   --input ${DIR}/${1}.fasta \
#   -o out_interpro/interproscan_${1}_output.tsv \
#   --formats tsv \
#   -verbose \
#   -incldepappl Pfam,ProSiteProfiles,ProSitePatterns,SMART,SUPERFAMILY,Gene3D \
#   --iprlookup --goterms

# The above command runs InterProScan using selected databases (including Pfam).
# The result is a TSV file (interproscan_output.tsv) containing all hits, including
# family/domain annotations with start-end coordinates.

# 3. Run cath-tools-genomescan


mkdir -p $(dirname $NEW_PATH)
../../cath-tools-genomescan/apps/cath-genomescan.pl -i ${1}.fasta -l ../../cath-tools-genomescan/data/funfam-hmm3-v4_2_0.lib -o $NEW_PATH
