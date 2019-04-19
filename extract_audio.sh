#!/bin/bash

# make an audio folder under Training and Validation, then run this script

for i in dataset/Training/Videos/*.mp4;
  do name=`echo $i | cut -d'.' -f1`;
  echo $name;
  ffmpeg -i "$i" -ab 160000 -ac 1 -ar 16000 -vn -f wav "${name/Videos/audio}.wav";
done

for i in dataset/Validation/Videos/*.mp4;
  do name=`echo $i | cut -d'.' -f1`;
  echo $name;
  ffmpeg -i "$i" -ab 160000 -ac 1 -ar 16000 -vn -f wav "${name/Videos/audio}.wav";
done