#!/bin/bash

container_image="webcam-ros-publisher"
 
echo "Building $container_image docker image."

sudo docker build . -t $container_image