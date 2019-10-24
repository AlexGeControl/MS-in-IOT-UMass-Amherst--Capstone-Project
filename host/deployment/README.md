# TF-Openpose Environment 

This Docker solution provides a running environment for tf-openpose with nvidia runtime

---

## Build

```shell
docker build -t intelli-train:frontend -f frontend.Dockerfile .
docker build -t intelli-train:backend -f backend.Dockerfile .
```

## Launch

The environment will:

- Camera Input: The instance will fetch video from **/dev/video0**
- Remote Desktop: The VNC server is running on port **5901**

Map your host device ID and port accordingly, then launch the instance using the following command:

```shell
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --device /dev/[HOST_VIDEO_DEVICE_ID]:/dev/video0 -p 5901:5901 -v $PWD/workspace:/workspace --name=frontend intelli-train:frontend
```

