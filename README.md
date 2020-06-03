## Head Pose Estimation

A cross platform (Linux and Windows) pose estimation algorithm for extreme rotations of head up to ~85 degrees using only RGB camera. 

This page contains the python code to robustly estimate the rotations of the head (3 DOF). The code used to regress facial keypoints for occluded facial area is obtained from: https://github.com/YadiraF/PRNet

The result is an end-to-end pipeline that seamlessly estimates very accurate facial pose for extreme rotations as well.

## Usage

1. Clone the repository

2. Download the PRN model from here: https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view and put it into PRNet/Data/net-data

3. Add the folder containing images in data/

4. Run the main.py file

