#!/bin/bash
export HOST_DEVICE_ID=/dev/video0
export HOST_VNC_PORT=5901
export HOST_BACKEND_PORT=60080

docker run --rm \
   --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
   --device $HOST_DEVICE_ID:/dev/video0 \
   -p $HOST_VNC_PORT:5901 \
   -v $PWD/workspace:/workspace \
   --name=frontend \
   intelli-train:frontend &

docker run --rm \
   --network=host \
   --name=database \
   mongo:latest &

docker run --rm \
   --network=host \
   -v $PWD/workspace/webapp:/app \
   --name=backend \
   intelli-train:backend &



