## Head Pose Estimation

Note: This page is still a bit under construction and will be subject to alteration..

A cross platform (Linux and Windows) pose estimation algorithm for extreme rotations of head up to ~85 degrees using only RGB camera. 

This page contains the python code to robustly estimate the rotations of the head (3 DOF). The code used to regress facial keypoints for occluded facial area is obtained from: https://github.com/YadiraF/PRNet

The result is an end-to-end pipeline that seamlessly estimates very accurate facial pose for extreme rotations as well.


## Pose Calculation Example: Roll
![Screenshot from 2020-06-08 12-25-50](https://user-images.githubusercontent.com/49958651/87858493-4600e900-c8fc-11ea-8d7e-1f0ab9c353c3.png)

## Usage

1. Usage example: python main.py -t "image_folder"

## Result
<p align="center">
  ![ezgif com-optimize (1)](https://user-images.githubusercontent.com/49958651/87860515-1528b000-c90c-11ea-9797-1fb15041576d.gif)
</p>

