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
        missing = np.array([])
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
            
            if success:
                self.top,self.left,self.bottom,self.right = bbox
                self.enlarge(bbox)
            
            try:
                image = image[self.left:self.right,self.top:self.bottom]
            except:
                missing = np.append(missing,int(self.frameNo))
                continue
            name = basename.split(".")[0]
            image_path = os.path.join(self.temp_path, name+"_"+str(self.top)+"_"+str(self.left)+".jpg")
            cv2.imwrite(image_path, image)
            
            if not bad_light:
                self.global_frame += 1
        if len(missing):
            np.savetxt(os.path.join(self.test_path,"missing.txt"),missing)
    
    def enlarge(self,bbox): ## Extract the facial area
        top,left,bottom,right = bbox
        center = [(bottom+top)//2, (right+left)//2]
        self.top,self.left,self.bottom,self.right = center[0]-self.roi//2,center[1]-self.roi//2,center[0]+self.roi//2,center[1]+self.roi//2
        self.top = max(0,self.top)
        self.left = max(0,self.left)
        self.bottom = max(0,self.bottom)
        self.right = max(0,self.right)
        
    def readKPT(self): ## structuring the kpt.txt file
        data = np.array([]).reshape(1,-1)
        with open (os.path.join(self.test_path,"kpt.txt"),"r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace("\n","")
            line = line.split(",")
            line = np.array(line).reshape(1,-1)
            line = line.astype(np.float)
            data = np.hstack([data,line])
        data = data.reshape(68,-1)
        np.savetxt(os.path.join(self.test_path,"kp.txt"),data)
        
    def getCenter(self,points):
        p0 = points[0]
        p1 = points[1]
        
        x = (p0[0] + p1[0])/2
        y = (p0[1] + p1[1])/2
        z = (p0[2] + p1[2])/2        
        return np.array([x,y,z])    
    
    def getAngle(self,point,type_):
        if type_ == "roll":        
            rise = point[1] 
            run = point[0]
        elif type_ == "yaw":
            rise = point[2] 
            run = point[0]
        elif type_=="pitch":
            rise = point[1] 
            run = point[2]
        theta = math.degrees(math.atan(rise/run))        
        return theta
    
    def getDist(self,points,type_):
        p0,p1 = points
        if type_=="roll":
            d = ((p0[0]-p1[0])**2 + (p0[2]-p1[2])**2)**0.5
        elif type_=="yaw":
            d = ((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)**0.5
        elif type_=="pitch":
            d = ((p0[0]-p1[0])**2 + (p0[2]-p1[2])**2)**0.5
        return d
            