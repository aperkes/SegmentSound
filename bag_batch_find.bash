#!/bin/bash
trap break INT

#Read through a list
bag_list=$1
kostas_path="birds@kostas-ap.seas.upenn.edu:/birds/aviary/data/tmp/sliced_bags/"
while IFS= read -r line
    do
        echo 'working on ' $line

        #file_path=$line
        file_path=${line%.bag}_files/${line%.bag}*.wv
        # Copy from kostas
        scp $kostas_path$file_path . 
        working_file=$(ls ${line%.bag}*.wv) 
        #working_file=${file_path#"/archive/marc/full_day_audio/"}
        echo 'running with: '$working_file
        bash find_songs.bash $working_file
        echo 'Deleting file'
        rm $working_file
    done < "$bag_list"


