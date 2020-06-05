
aviary_day=$1
#working_dir='~/Documents/Scripts/SegmentSound/working_dir/'
img_dir=$aviary_day'images/'
wav_dir=$aviary_day'wavs/'
av_dir=$aviary_day'combos/'

mkdir $av_dir

for file in $img_dir*.png
do
    echo 'Working on' "$file"
    clip_base=${file%.png}
    clip_base=$(basename "$file")
    img_file=$clip_base
    wav_file=${clip_base%.png}.wav
    av_file=${clip_base%.png}.mp4
    echo $img_dir$img_file $wav_dir$wav_file 
    echo $av_dir$av_file

    ffmpeg -loop 1 -i $img_dir$img_file -i $wav_dir$wav_file -shortest -c:a aac -strict -2 -c:v h264 -vf "scale=w=640:h=480:force_original_aspect_ratio=1,pad=640:480:(ow-iw)/2:(oh-ih)/2:color=black" $av_dir$av_file 
done
