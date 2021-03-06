#!/bin/bash
python3 launch-frontend.py \
    --camera=0 \
    --input-mode=left \
    --reference=set-00.json \
    --prep-time=20.0 \
    --resize=432x368 \
    --resize-out-ratio=5.0 \
    --model=mobilenet_v2_large \
    # --show-process=True \
    # TODO: enable TensorRT
    # --tensorrt=False