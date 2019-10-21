#!/bin/bash
export HOST_DEVICE_ID=/dev/video1
export HOST_VNC_PORT=5901

docker run --rm \
   --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
   --device $HOST_DEVICE_ID:/dev/video0 \
   -p $HOST_VNC_PORT:5901 \
   -v $PWD/workspace:/workspace \
   --name=frontend \
   intelli-train:frontend



