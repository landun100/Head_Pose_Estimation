import numpy as np
import cv2
import glob
import os
import math
import shutil
import face_detector

class Util():
    def __init__(self, test_path="", image_format="jpeg", roi=200):
        self.test_path = test_path
        self.image_format = image_format
        self.roi = roi
        self.net = cv2.dnn.readNetFromCaffe("./detection_models/deploy.prototxt.txt", "./detection_models/res10_300x300_ssd_iter_140000.caffemodel")
        self.face_detector = face_detector.FaceDetector()
        self.global_frame = 0
        self.temp_path = os.path.join(test_path, "temp")
        
        if os.path.isdir(self.temp_path):
            shutil.rmtree(self.temp_path)
            
        os.mkdir(self.temp_path)

    def processImage(self,bad_light=False):
        '''
        Detects Facial BBox and predicts if not detected.        

        Parameters
        ----------
        bad_light : Optional argument. Will swith to HAAR based classifier if enabled
        Usage: When there is body movement in the frames AND BADLIGHT

        Returns
        -------
        None.

        '''
        frames = glob.glob(os.path.join(self.test_path,"*"+self.image_format))
        labels = {}
        dummy = np.zeros((self.roi,self.roi,3))
        for frame in frames: ## Sort Images
            frame = os.path.split(frame)[-1][5:-5].zfill(5)
            new = {frame:str(int(frame))}
            labels.update(new)

        for key in sorted(labels):
            self.frameNo = labels[key]
            self.file = os.path.join(self.test_path,"frame"+self.frameNo+"."+self.image_format)  
            basename = os.path.basename(self.file)

            image = cv2.imread(self.file)
            success,bbox = self.face_detector.detect(image,self.global_frame)            
            name = basename.split(".")[0]
            
            if success:
                self.top,self.left,self.bottom,self.right = bbox
                self.enlarge(bbox)
            
            try:
                image = image[self.left:self.right,self.top:self.bottom]
            except:
                dummy_path = os.path.join(self.temp_path, name+"_"+str(1000)+"_"+str(1000)+".jpg") ## Saving blank image, as saving image as it is without BBox can create false positives
                cv2.imwrite(dummy_path, dummy)                
                continue

            image_path = os.path.join(self.temp_path, name+"_"+str(self.top)+"_"+str(self.left)+".jpg")
            cv2.imwrite(image_path, image)
            
            if not bad_light:
                self.global_frame += 1

    
    def enlarge(self,bbox): ## Extract the facial area
        top,left,bottom,right = bbox
        center = [(bottom+top)//2, (right+left)//2]
        self.top,self.left,self.bottom,self.right = center[0]-self.roi//2,center[1]-self.roi//2,center[0]+self.roi//2,center[1]+self.roi//2
        self.top = max(0,self.top)
        self.left = max(0,self.left)
        self.bottom = max(0,self.bottom)
        self.right = max(0,self.right)
        

        

            