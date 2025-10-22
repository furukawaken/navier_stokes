#!/bin/bash

#conda run -n simulation python visualize.py
python visualize_uvp.py

ffmpeg -framerate 5 -i ./fig/uvp/snapshot_%05d.png -c:v libx264 -pix_fmt yuv420p ./movie/uvp/uvp.mp4