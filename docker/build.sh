#!/bin/bash

docker buildx build -f Dockerfile \
    -t "slora:latest" \
    --platform linux/amd64 \
    --build-arg USER_ID=1000 \
    --build-arg USER_NAME=slora \
    --build-arg GROUP_ID=1000 \
    --build-arg GROUP_NAME=slora \
    --load .