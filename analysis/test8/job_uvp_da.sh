#!/bin/bash

#conda run -n simulation python visualize.py
python visualize_uvp_da.py

ffmpeg -framerate 5 -i ./fig/uvp_da/snapshot_%05d.png -c:v libx264 -pix_fmt yuv420p ./movie/uvp_da/uvp.mp4