## Head Pose Estimation

Note: This page is still a bit under construction and will be subject to alteration..

A cross platform (Linux and Windows) pose estimation algorithm for extreme rotations of head up to ~85 degrees using only RGB camera. 

This page contains the python code to robustly estimate the rotations of the head (3 DOF). The code used to regress facial keypoints for occluded facial area is obtained from: https://github.com/YadiraF/PRNet

The result is an end-to-end pipeline that seamlessly estimates very accurate facial pose for extreme rotations as well.

## Usage

1. Usage example: python main.py -t "image_folder"

