
## Take a wav as input, spit out a file with all the times of songs

input_wv=$(basename "$1")
working_dir='./working_dir/'
png_directory=$working_dir${input_wv%.wv}
input_wav=${input_wv%.wv}
echo $png_directory
echo input_wav


echo "Making directory"
mkdir $png_directory
mkdir $png_directory/images
mkdir $png_directory/wavs

echo "Converting to .wavs"
#ffmpeg -nostdin -i $input_wv $input_wav
ffmpeg -nostdin -i $1 -f segment -segment_time 3600 $working_dir$input_wav'_'%03d.wav

echo "Detecting songs..."
for file in $working_dir$input_wav*.wav
do
    echo 'Running on' "$file"
    python ./detect_sound.py --wav "$file" --out_dir $png_directory/images/
    echo 'Deleteing' "$file"
    rm $file
done

echo "Classifying..."
python ./SongClassifier/predict_song.py $png_directory

echo "not Deleting pngs"
#rm $png_directory/images/*

echo "Not Deleting any wavs"
#rm $png_directory/wavs/*

#echo "Deleting .wav"
#rm $input_wav
