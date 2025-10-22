#!/bin/bash

#conda run -n simulation python visualize.py
python visualize_streamline_da.py

ffmpeg -framerate 5 -i ./fig/streamline_da/snapshot_%05d.png -c:v libx264 -pix_fmt yuv420p ./movie/streamline_da/streamline.mp4