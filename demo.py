import numpy as np
import os
from glob import glob
from skimage.io import imread
from skimage.transform import rescale, resize
from time import time

from api import PRN

class PRNet():
    def __init__(self, test_path):
        test_path = os.path.abspath(test_path)
        self.main(test_path)

    def main(self, test_path, gpu="0", isDlib=True):
        # ---- init PRN
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu # GPU number, -1 for CPU
        prn = PRN(is_dlib = isDlib)
    
        # ------------- load data
        types = ('*.jpg')
        image_path_list= []
        
        temp_path = os.path.join(test_path, "temp")
        
        for files in types:
            image_path_list.extend(glob(os.path.join(temp_path, files)))
        total_num = len(image_path_list)
        
        kpt_frames = []
        
        image_paths = []
        maxFrame = 0

        for image_path in os.listdir(temp_path):
            if image_path.endswith(".jpg"):
                image_paths.append(os.path.join(temp_path, image_path))
                
                frameNum = int(image_path.split('_')[0].replace("frame", ""))
                
                if frameNum > maxFrame:
                    maxFrame = frameNum
                    
        frameNum = 0
        sorted_image_paths = []
        
        while frameNum <= maxFrame:
            for image_path in image_paths:
                name = os.path.split(image_path)[-1]
                
                if name.startswith("frame" + str(frameNum) + "_"):
                    sorted_image_paths.append(image_path)
                    break
            
            frameNum += 1
        
        for image_path in sorted_image_paths:
            name = os.path.split(image_path.strip())[-1][:-4]
    
            # read image
            image = imread(image_path)
            [h, w, c] = image.shape
            if c>3:
                image = image[:,:,:3]
                
            max_size = max(image.shape[0], image.shape[1])
            
            if max_size> 1000:
                image = rescale(image, 1000./max_size)
                image = (image*255).astype(np.uint8)
                
            pos = prn.process(image) # use dlib to detect face
            
            image = image/255.
            
            kpts = []
            
            if pos is not None:
                # get landmarks
                kpts = prn.get_landmarks(pos)
            else:
                index = 0
                
                while index < 68:
                    kpts.append([1000.0,1000.0,1000.0])
                    index += 1
                      
            kpt_frames.append(kpts)
            
        kpt_path = os.path.join(test_path, "kpt.txt")
            
        pt_index = 0
        pt_count = len(kpt_frames[0])

        lines = []

        while pt_index < pt_count:
            line = ""
            for frame in kpt_frames:
                pt = frame[pt_index]
                line += str(pt[0]) + "," + str(pt[1]) + "," + str(pt[2]) + ", "
            
            line = line[:-2]
            lines.append(line + "\n")
            
            pt_index += 1

        with open(kpt_path, 'w') as kpt_file:
            kpt_file.writelines(lines)