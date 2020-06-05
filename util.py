import numpy as np
import cv2
import glob
import os
import math
import shutil
import face_detector


class Util():
    def __init__(self,image_folder="",save_folder="./temp/",image_format="jpeg",roi=200):
        self.image_folder = image_folder
        self.save_folder = save_folder
        self.image_format = image_format
        self.roi = roi
        self.net = cv2.dnn.readNetFromCaffe("./detection_models/deploy.prototxt.txt", "./detection_models/res10_300x300_ssd_iter_140000.caffemodel")
        self.clahe = False
        self.face_detector = face_detector.FaceDetector()
        self.global_frame = 0
        self.arrange()
        if os.path.isdir(save_folder):
            shutil.rmtree("temp")
            os.mkdir("temp")
        else:
            os.mkdir("./temp")


    def processImage(self,movement=False):
        for self.file in sorted(glob.glob(os.path.join(self.image_folder,"*"+self.image_format))):
            basename = os.path.basename(self.file)
            self.frameNo = basename.split(".")[0][5:]
            print(self.frameNo)
            image = cv2.imread(self.file)
            image.shape
            success,bbox = self.face_detector.detect(image,self.global_frame)
            if success:
                self.top,self.left,self.bottom,self.right = bbox
                self.enlarge(bbox)
            # top,left,bottom,right = self.ssd(image)
            # top,left,bottom,right = self.getHeadJoint()
            
            image = image[self.left:self.right,self.top:self.bottom]
            name = basename.split(".")[0]
            cv2.imwrite(self.save_folder+name+"_"+str(self.top)+"_"+str(self.left)+"_.jpg", image)
            if not movement:
                self.global_frame+=1
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
                if self.clahe:
                    return        
        try:
            center = [(self.bottom+self.top)//2, (self.right+self.left)//2]
        except:
            if not self.clahe:
                self.clahe = True
                gray = self.CLAHE(image)
                self.ssd(gray)
                center = [(self.bottom+self.top)//2, (self.right+self.left)//2]
            else:
                pass

        top,left,bottom,right = center[0]-self.roi//2,center[1]-self.roi//2,center[0]+self.roi//2,center[1]+self.roi//2
        return top,left,bottom,right
    
    def enlarge(self,bbox):
        top,left,bottom,right = bbox
        center = [(bottom+top)//2, (right+left)//2]
        self.top,self.left,self.bottom,self.right = center[0]-self.roi//2,center[1]-self.roi//2,center[0]+self.roi//2,center[1]+self.roi//2
        self.top = max(0,self.top)
        self.left = max(0,self.left)
        self.bottom = max(0,self.bottom)
        self.right = max(0,self.right)
        
    
    def estimateFrames(self,frames,missing_frames,approx=3,curve=0): ## Estimate angle for frames with missing facial keypoints
        last_frame_missing = False
        gaps = []
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
                    try:
                        prevFrame+=frames[np.int_(frames[:,0]) == estimate[0]-1].reshape(-1,4)[0]
                        nextFrame+=frames[np.int_(frames[:,0]) == estimate[-1]+1].reshape(-1,4)[0]
                    except:
                        last_frame_missing = True
                        print("Missing facial points on last frame")
                        break
            if last_frame_missing: ## TODO: Remove this
                break

                
            prevFrame = prevFrame/approx
            nextFrame = nextFrame/approx
            
            est = 0.5*prevFrame + 0.5*nextFrame
            est = np.ones((len(estimate),1))*est
            
            
            if len(estimate) > 4:
                momentum = prevFrame - frames[np.int_(frames[:,0]) == estimate[0]-approx-curve].reshape(-1,4)[0]
                # next_momentum = nextFrame - frames[np.int_(frames[:,0]) == estimate[-1]+approx-curve].reshape(-1,4)[0]              
        
                forward = np.matmul((np.array((list(range(1,len(estimate)//2+1))),dtype=np.float64).reshape(-1,1)/(len(estimate)//2)),momentum)

                backward = np.matmul((np.array((list(range(len(estimate)//2,1-1,-1))),dtype=np.float64).reshape(-1,1)/(len(estimate)//2)),momentum)
                
                if len(estimate)%2:
                    forward = np.vstack([forward,forward[-1]])
                    # backward = np.vstack([backward,backward[-1]])          
                    est[:len(estimate)//2+1] = est[:len(estimate)//2+1]+forward
                    est[len(estimate)//2+1:] = est[len(estimate)//2+1:]+backward
                else:
                    est[:len(estimate)//2] = est[:len(estimate)//2]+forward
                    est[len(estimate)//2:] = est[len(estimate)//2:]+backward
            
            estimate = np.array(estimate)
            gaps.append(estimate)
            estimated = np.hstack([estimate.reshape(-1,1), est[:,1:]])
            estimated_frames = np.vstack([estimated_frames,estimated])         
                

            frame+=1
            estimate = [missing_frames[frame]]
            prevFrame = np.zeros((1,4))
            nextFrame = np.zeros((1,4))
                
        total_frames = np.vstack([frames, estimated_frames])
        return total_frames, np.array(gaps).reshape(1,-1)
    
    
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
    
    def CLAHE(self,image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)        
        lab_planes = cv2.split(lab)        
        clahe = cv2.createCLAHE(clipLimit=1,tileGridSize=(100,100))        
        lab_planes[0] = clahe.apply(lab_planes[0])        
        lab = cv2.merge(lab_planes)        
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)   
        
        return image
    
    def show(self,image):
        cv2.imshow("img",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    def arrange(self):
        for filename in glob.glob(os.path.join(self.image_folder,"*"+self.image_format)):
            basename = os.path.basename(filename)
            frameNo = basename.split(".")[0][5:]
            frameNo = str(frameNo).zfill(5)
            path = os.path.split(filename)[0]
            new_filename = os.path.join(path,"frame"+frameNo+".jpeg")
            os.rename(filename, new_filename)
            
        for i,filename in enumerate(sorted(glob.glob(os.path.join(self.image_folder,"*"+self.image_format)))):
            frameNo = str(i).zfill(5)
            path = os.path.split(filename)[0]
            new_filename = os.path.join(path,"frame"+frameNo+".jpeg")
            os.rename(filename, new_filename)
            
            


        