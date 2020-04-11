
## Take a wav as input, spit out a file with all the times of songs

input_wv=$(basename "$1")
working_dir='./working_dir/'
png_directory=$working_dir${input_wv%.wv}
input_wav=${input_wv%wv}wav
echo $png_directory
echo input_wav


echo "Making directory"
mkdir $png_directory
mkdir $png_directory/images
echo "Converting to .wav"
ffmpeg -i $input_wv $input_wav

echo "Detecting songs..."
python ./detect_sound.py $input_wav $png_directory/images/

echo "Classifying..."
python ./SongClassifier/predict_song.py $png_directory

echo "Deleting pngs"
rm $png_directory/images/*

echo "Deleting .wav"
rm $input_wav
