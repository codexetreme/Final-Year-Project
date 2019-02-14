#!/bin/bash
# Args: 
#   $1: the dataset folder
#   $2: location of the folder where the concatenated corpus.txt will be saved
# WARNING: this script can take some time to run....
for i in `ls $1`
do
    i=$1/$i
    cat $i | sed -e 's/-LRB-/(/g' -e 's/-RRB-/)/g' >> $2/corpus.txt
done