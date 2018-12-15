sudo rm  Images/*
sudo rm  ImagesOutput/*

ffmpeg -i VideoInput/video.mp4 -vf fps=20 Images/image%01d.jpg -hide_banner

cd Images/
o=$(ls -1 --file-type | grep -v '/$' | wc -l)
export o
cd ..
python3 Inference.py
ffmpeg -framerate 20 -i ImagesOutput/image%01d.jpg VideoOutput/video.mp4
cd VideoOutput/
m=$(ls -1 --file-type | grep -v '/$' | wc -l)
mv video.mp4 video$m.mp4 

