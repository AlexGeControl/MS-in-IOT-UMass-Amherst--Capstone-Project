#!/bin/bash
file /lib/systemd/system/docker.service
file /lib/systemd/system/docker.socket
systemctl unmask docker.service
systemctl unmask docker.socket
systemctl start docker.service
systemctl status docker

export HOST_DEVICE_ID=/dev/video0
export HOST_VNC_PORT=5901

docker run --rm \
   --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
   --device $HOST_DEVICE_ID:/dev/video0 \
   -p $HOST_VNC_PORT:5901 \
   -v $PWD/workspace:/workspace \
   --name=frontend \
   intelli-train:frontend



