#!/bin/bash
python3 generate-reference.py \
    --video=set-00.mp4 \
    --resize=432x368 \
    --resize-out-ratio=5.0 \
    --model=mobilenet_v2_large \
    # --show-process=True \
    # TODO: enable TensorRT
    # --tensorrt=False