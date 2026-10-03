#!/bin/sh
# Export frames for reviewing the steam effect: a zoomed contact sheet over one
# loop, the full frame, and the loop seam (last frame next to the first).
set -e
src=${1:-output/night_study.mp4}
out=${2:-output/review}
mkdir -p "$out"
ffmpeg -y -loglevel error -i "$src" -vf "select='not(mod(n\,12))',crop=260:320:760:500,scale=390:-1,tile=6x2" -frames:v 1 "$out/steam_sheet.png"
ffmpeg -y -loglevel error -i "$src" -vf "select='eq(n\,60)'" -frames:v 1 "$out/full_frame.png"
ffmpeg -y -loglevel error -i "$src" -vf "select='gte(n\,140)+lte(n\,3)',crop=260:320:760:500,tile=8x1" -frames:v 1 "$out/steam_seam.png"
ffmpeg -y -loglevel error -i "$src" -vf "select='between(n\,40\,51)',crop=260:320:760:500,scale=390:-1,tile=6x2" -frames:v 1 "$out/steam_motion.png"
