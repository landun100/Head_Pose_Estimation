import numpy as np
import cv2
import glob
import os
import math
import shutil


class Util():
    def __init__(self,image_folder="",save_folder="./temp/",image_format="jpeg",roi=200):
        self.image_folder = image_folder
        self.save_folder = save_folder
        self.image_format = image_format
        self.roi = roi
        self.net = cv2.dnn.readNetFromCaffe("./detection_models/deploy.prototxt.txt", "./detection_models/res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.isdir(save_folder):
            shutil.rmtree("temp")
            os.mkdir("temp")
        else:
            os.mkdir("./temp")


    def processImage(self):
        num_images = len(glob.glob(os.path.join(self.image_folder,"*"+self.image_format)))
        for x in range(num_images):
            self.file = os.path.join(self.image_folder,"frame"+str(x)+"."+self.image_format)
            basename = os.path.basename(self.file)
            self.frameNo = basename.split(".")[0][5:]
            image = cv2.imread(self.file)
            top,left,bottom,right = self.ssd(image)
            # top,left,bottom,right = self.getHeadJoint()
            
            image = image[left:right,top:bottom]
            name = basename.split(".")[0]
            cv2.imwrite(self.save_folder+name+"_"+str(top)+"_"+str(left)+"_.jpg", image)
        return "../temp/"

           
    def getHeadJoint(self): ## If using Kinect or any other method to get head joint
        filename = self.image_folder+"skele_color_depth"+self.frameNo+".txt"
        with open(filename,"r") as f:
            data = f.readlines()
        for line in data:
            line = line.replace("\n","").replace(",","")
            coords = line.split(" ")
            head_joint = ([int(float(coords[2])), int(float(coords[3]))])
            break
        (top,left,bottom,right) = head_joint[0]-self.roi//2,head_joint[1]-self.roi//2,head_joint[0]+self.roi//2,head_joint[1]+self.roi//2
        return top,left,bottom,right
    
    def weightedAvergage(self,pose,total_frames,beta=0.9):
        prev_pose = pose[np.int_(pose[:,0]) == 0].reshape(-1,4)[0]
        average = np.array(prev_pose).reshape(-1,4)
        for x in range(1,total_frames):
            try:
                current_pose = pose[np.int_(pose[:,0]) == x].reshape(-1,4)[0]
            except:
                continue
            avg = np.hstack([x,beta*prev_pose[1:] + (1-beta)*current_pose[1:]])
            average = np.vstack([average, avg])
            prev_pose = avg
        return average
            
    
    def ssd(self,image):   
        (h,w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (self.top,self.left,self.bottom,self.right) = box.astype("int")
                self.width = self.bottom-self.top
                self.height = self.right-self.left
        
        center = [(self.bottom+self.top)//2, (self.right+self.left)//2]        
        top,left,bottom,right = center[0]-self.roi//2,center[1]-self.roi//2,center[0]+self.roi//2,center[1]+self.roi//2
        return top,left,bottom,right
    
    def estimateFrames(self,frames,missing_frames,approx=3,curve=0): ## Estimate angle for frames with missing facial keypoints
        missing_frames = sorted(missing_frames)
        missing_frames+=[0]
        frame=0
        prevFrame = np.zeros((1,4))
        nextFrame = np.zeros((1,4))
        estimated_frames = np.array([]).reshape(-1,4)
        estimate = [missing_frames[frame]]
        while True:
            if frame == len(missing_frames)-1:
                break
            if missing_frames[frame+1] - missing_frames[frame] == 1:
                estimate.append(missing_frames[frame+1])
                frame+=1
                continue
                    
            for x in range(approx):
                try:
                    prevFrame+=frames[np.int_(frames[:,0]) == estimate[0]-(x+1)].reshape(-1,4)[0]
                    
                    try:
                        nextFrame+=frames[np.int_(frames[:,0]) == estimate[-1]+(x+1)].reshape(-1,4)[0]
                    except:
                        t1=x-1
                        nextFrame+=frames[np.int_(frames[:,0]) == estimate[-1]+(t1+1)].reshape(-1,4)[0]
                    
                except:
                    prevFrame+=frames[np.int_(frames[:,0]) == estimate[0]-1].reshape(-1,4)[0]
                    nextFrame+=frames[np.int_(frames[:,0]) == estimate[-1]+1].reshape(-1,4)[0]

                
            prevFrame = prevFrame/approx
            nextFrame = nextFrame/approx
            
            est = 0.5*prevFrame + 0.5*nextFrame
            est = np.ones((len(estimate),1))*est
            
            
            if len(estimate) > 4:
                momentum = prevFrame - frames[np.int_(frames[:,0]) == estimate[0]-approx-curve].reshape(-1,4)[0]
                # next_momentum = nextFrame - frames[np.int_(frames[:,0]) == estimate[-1]+approx-curve].reshape(-1,4)[0]              
        
                forward = np.matmul((np.array((list(range(1,len(estimate)//2+1)))).reshape(-1,1)/(len(estimate)//2)),momentum)

                backward = np.matmul((np.array((list(range(len(estimate)//2,1-1,-1)))).reshape(-1,1)/(len(estimate)//2)),momentum)
                
                if len(estimate)%2:
                    forward = np.vstack([forward,forward[-1]])
                    # backward = np.vstack([backward,backward[-1]])          
                    est[:len(estimate)//2+1] = est[:len(estimate)//2+1]+forward
                    est[len(estimate)//2+1:] = est[len(estimate)//2+1:]+backward
                else:
                    est[:len(estimate)//2] = est[:len(estimate)//2]+forward
                    est[len(estimate)//2:] = est[len(estimate)//2:]+backward
            
            estimate = np.array(estimate)
            estimated = np.hstack([estimate.reshape(-1,1), est[:,1:]])
            estimated_frames = np.vstack([estimated_frames,estimated])         
                

            frame+=1
            estimate = [missing_frames[frame]]
            prevFrame = np.zeros((1,4))
            nextFrame = np.zeros((1,4))
                
        total_frames = np.vstack([frames, estimated_frames])
        return total_frames
    
    
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