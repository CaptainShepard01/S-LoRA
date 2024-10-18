#!/bin/bash

docker buildx build -f Dockerfile \
    -t "registry.rcp.epfl.ch/sacs_balykov/slora:latest" \
    --platform linux/amd64 \
    --build-arg USER_ID=269883 \
    --build-arg USER_NAME=balykov \
    --build-arg GROUP_ID=30133 \
    --build-arg GROUP_NAME=sacs \
    --load .