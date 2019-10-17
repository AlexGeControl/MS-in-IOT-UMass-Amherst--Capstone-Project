#!/bin/bash
python3 launch-frontend.py \
    --resize=432x368 \
    --model=mobilenet_v2_small \
    # --show-process=True \
    # TODO: enable TensorRT
    # --tensorrt=False