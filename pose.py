import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import util
import warnings
warnings.filterwarnings("ignore")
import shutil


class Pose():
    def __init__(self,folder):
        '''        
        Calculates the roll, pitch yaw based on facial keypoints.
        
        Angles are calculate based on slope of the line between two keypoints chosen.
        Chosen Keypoints:
            Roll : outside points of the eyes
            Pitch :  point on jaw on either side
            Yaw: point on the lip and point lying on the center of the line passing through two points used for pitch
        
        Results saved into pose.txt file

        '''
        self.folder = folder
        ut = util.Util()
        keypoints = np.loadtxt(os.path.join(self.folder, "kp.txt"))
        pose = np.array([]).reshape(-1,3)
        for k in range(keypoints.shape[1]//3):
            kp = keypoints[:,k*3:k*3+3]
            
            ## ROll
            eye0 = kp[36]
            eye1 = kp[45]
            
            if eye0[0] == 1000: ## If no keypoints
                euler = np.array([1000.,1000.,1000.]).reshape(-1,3)
                pose = np.vstack([pose,euler])
                continue           
                
            points = np.array([eye0,eye1])
            eyeCenter = ut.getCenter(points)
            
            kp_roll = kp-eyeCenter
            
            eye0 = kp_roll[36]
            eye1 = kp_roll[45]
            points = np.array([eye0,eye1])
            eyeDist = ut.getDist(points,type_="roll")
            
            x = eyeDist/2
            
            eye0_x = x*(abs(eye0[0])/eye0[0])
            eye0_y = eye0[1]
            eye0_z = 0.
            
            eye1_x = x*(abs(eye1[0])/eye1[0])
            eye1_y = eye1[1]
            eye1_z = 0.
            
            eye0 = np.array([eye0_x,eye0_y,eye0_z])
            eye1 = np.array([eye1_x,eye1_y,eye1_z])
            
            
            roll = ut.getAngle(eye0,type_="roll")
            
     
            ## YAW
            jaw0 = kp[3]
            jaw1 = kp[13]
            points = np.array([jaw0,jaw1])
            jawCenter = ut.getCenter(points)
            
            kp_yaw = kp-jawCenter
            
            jaw0 = kp_yaw[3]
            jaw1 = kp_yaw[13]
            points = np.array([jaw0,jaw1])
            jawDist = ut.getDist(points,type_="yaw")
            
            x = jawDist/2
            
            jaw0_x = x*(abs(jaw0[0])/jaw0[0])
            jaw0_y = 0.
            jaw0_z = jaw0[2]
            
            jaw1_x = x*(abs(jaw1[0])/jaw1[0])
            jaw1_y = 0.
            jaw1_z = jaw1[2]
            
            jaw0 = np.array([jaw0_x,jaw0_y,jaw0_z])
            jaw1 = np.array([jaw1_x,jaw1_y,jaw1_z])
            
            yaw = ut.getAngle(jaw0,type_="yaw")
            yaw = yaw*-1
            
            
            ## PITCH
            jaw0 = kp[3]
            jaw1 = kp[13]
            points = np.array([jaw0,jaw1])
            jawCenter = ut.getCenter(points)
            
            kp_pitch = kp-jawCenter
            
            lip = kp_pitch[62]
            points = np.array([[0,0,0],lip])
            hypCenter = ut.getCenter(points)
            
            kp_pitch = kp_pitch-hypCenter
            
            lip = kp_pitch[62]
            jaw0 = kp_pitch[3]
            jaw1 = kp_pitch[13]
            points = np.array([jaw0,jaw1])
            jawCenter = ut.getCenter(points)
            
            points = np.array([lip,jawCenter])
            hypDist = ut.getDist(points,type_="pitch")
            
            z = hypDist/2
            
            lip_x = 0.
            lip_y = lip[1]
            lip_z = z*(abs(lip[2])/lip[2])
            
            jawCenter_x = 0.
            jawCenter_y = jawCenter[1]
            jawCenter_z = z*(abs(jawCenter[2])/jawCenter[2])
            
            lip = np.array([lip_x,lip_y,lip_z])
            jawCenter = np.array([jawCenter_x,jawCenter_y,jawCenter_z])
            
            pitch = ut.getAngle(lip,type_="pitch")

            
            euler = np.array([roll,pitch,yaw]).reshape(-1,3)
            pose = np.vstack([pose,euler])            
        
        np.savetxt(os.path.join(self.folder,"pose.txt"),pose)
        

    def regress(self,beta=0.7,curve=0):
        '''
        Estimates the in-between missing frames.
        
        Beta : Smoothening factor used for exponential weighted average.
        curve : Will approximate the slope for missing frames based on this parameter provided.
                Higher the value, higher will be the slope estimated
                
        Results saved as: 
            pose_estimated_raw.txt: pose with in-between missing frame prediction
            pose_estimated_smooth: applying weighted average to pose_estimated_raw


        '''
        data = np.loadtxt(os.path.join(self.folder,"pose.txt")) ## Read the pose.txt file
        data_labelled = np.hstack([np.array(range(data.shape[0])).reshape(-1,1),data])
        data = data_labelled
        
        missing = data_labelled[data_labelled[:,1]==1000]
        if len(missing):
            missing = missing[:,0].reshape(-1,1)
    
            data_labelled = np.delete(data_labelled,(missing),0)
            
            regressed_pose = self.estimateFrames(data_labelled, missing,curve=curve) ## Estimate in-between missing frames
            regressed_pose = regressed_pose[np.argsort(regressed_pose[:, 0])]          
            
            average = self.weightedAvergage(regressed_pose,beta=beta) ## Exponential weighted avergaging      
           
            fill = np.ones((len(missing),1))*1000
            missing = np.hstack([missing,fill,fill,fill])
                
            not_regressed = np.array([]).reshape(-1,4)
            for x in range(len(missing)):
                if missing[x,0] not in regressed_pose[:,0]:
                    not_regressed = np.vstack([not_regressed,missing[x]])         
        
            regressed_pose = np.vstack([regressed_pose, not_regressed]) ## Add the missing (failed to regress) frames
            average = np.vstack([average,not_regressed])
            
            if os.path.isfile(os.path.join(self.folder,"missing.txt")): ## Add the initial missing frames
                missing_init = np.loadtxt(os.path.join(self.folder,"missing.txt")).reshape(-1,1)
                fill = np.ones((len(missing_init),1))*1000
                missing_init = np.hstack([missing_init,fill,fill,fill])
                last_missing = missing_init[-1,0]+1
                
                regressed_pose[:,0]+= last_missing
                average[:,0]+= last_missing
                data[:,0]+= last_missing
                
                regressed_pose = np.vstack([regressed_pose,missing_init])  
                average = np.vstack([average,missing_init])  
                data = np.vstack([data,missing_init])  
            
            regressed_pose = regressed_pose[np.argsort(regressed_pose[:, 0])]
            average = average[np.argsort(average[:, 0])] 
            data = data[np.argsort(data[:, 0])] 
            
            np.savetxt(os.path.join(self.folder,"pose_estimated_raw.txt"),regressed_pose[:,1:])
            np.savetxt(os.path.join(self.folder,"pose_estimated_smooth.txt"),average[:,1:])
            np.savetxt(os.path.join(self.folder,"pose.txt"),data[:,1:])
 

    def estimateFrames(self,frames,missing_frames,approx=3,curve=3): ## Estimate angle for frames with missing facial keypoints
        last_frame_missing = False
        first_frame_missing = False
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
            if missing_frames[frame+1] - missing_frames[frame] == 1: ## Separate Contiguous missing frames
                estimate.append(missing_frames[frame+1])
                frame+=1
                continue
                    
            for x in range(approx): ## Take average over last observed frames (Higher the value of approx the more reliable result) since last observed frame can have false readings
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
            
            if last_frame_missing:
                break

                
            prevFrame = prevFrame/approx ## Average of frames observed before in-between missing frames
            nextFrame = nextFrame/approx ## Average of frames observed after in-between missing frames
            
            est = 0.5*prevFrame + 0.5*nextFrame ## Estimated slope 
            est = np.ones((len(estimate),1))*est
            
            
            if len(estimate) > 4: ## If length of missing frames is less than 4, then no ramp function is applied
                try:
                    momentum = prevFrame - frames[np.int_(frames[:,0]) == estimate[0]-approx-curve].reshape(-1,4)[0] ## angle values used to estimate the missing frames
                    # next_momentum = nextFrame - frames[np.int_(frames[:,0]) == estimate[-1]+approx-curve].reshape(-1,4)[0]  
                except:
                    frame+=1
                    estimate = [missing_frames[frame]]
                    prevFrame = np.zeros((1,4))
                    nextFrame = np.zeros((1,4))
                    continue
        
                forward = np.matmul((np.array((list(range(1,len(estimate)//2+1))),dtype=np.float64).reshape(-1,1)/(len(estimate)//2)),momentum) ## ramp

                backward = np.matmul((np.array((list(range(len(estimate)//2,1-1,-1))),dtype=np.float64).reshape(-1,1)/(len(estimate)//2)),momentum)
                
                if len(estimate)%2:
                    forward = np.vstack([forward,forward[-1]])
                    # backward = np.vstack([backward,backward[-1]])          
                    est[:len(estimate)//2+1] = est[:len(estimate)//2+1]+forward ## first missing frames to the mid frame of missing frames
                    est[len(estimate)//2+1:] = est[len(estimate)//2+1:]+backward ## mid frame of missing frames to last missing frame
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
                
        total_frames = np.vstack([frames, estimated_frames]) ## All frames with estimation
        return total_frames#, np.array(gaps).reshape(1,-1)
    
    def weightedAvergage(self,pose,beta=0.9): ## Exponential weighted average
        total_frames = pose.shape[0]
        prev_pose = pose[0]
        average = np.array(prev_pose).reshape(-1,4)
        for x in range(1,total_frames):            
            current_pose = pose[x]
            avg = np.hstack([pose[x,0],beta*prev_pose[1:] + (1-beta)*current_pose[1:]])
            average = np.vstack([average, avg])
            prev_pose = avg
        return average