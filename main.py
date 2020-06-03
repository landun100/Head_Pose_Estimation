import sys
sys.path.insert(1, "./PRNet/")
import util as ut
import os
import demo
import pose
import video_writer

data = False ## Set True if facial keypoints already captured

data_folder = "Ari" ## Name of the folder to get the head pose for
img_format="jpeg"

image_folder = "./data/"+data_folder+"/"
save_folder = "./Results/"+data_folder+"/"
    
if not data:
    ut = ut.PreProcess(image_folder,roi=200, image_format=img_format)
    folder = ut.processImage()
    
    os.chdir("./PRNet/")
    demo.PRNet(folder,save_folder)
    os.chdir("..")
    
head_pose = pose.Pose(save_folder)
head_pose.regress(estimate=True,save_plot=True,beta=0.7,curve=8)

video_writer.Video(image_folder, save_folder,plot=True)