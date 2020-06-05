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
        self.folder = folder
        ut = util.Util()
        for filename in glob.glob(self.folder+"*kpt.txt"):            
            basename = os.path.basename(filename)
            frame = basename.split("_")[0]
            kp = np.loadtxt(filename,dtype=np.float64)
            
            ## ROll
            eye0 = kp[36]
            eye1 = kp[45]
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
            np.savetxt(self.folder+"pose"+frame[5:]+".txt",euler)
            
        shutil.rmtree("temp")
        
        self.missing = glob.glob(self.folder+"*failed.txt")
        missing_frames = []
        for pose in self.missing:
            basename = os.path.basename(pose)
            frameNo = basename.split("_")[0][5:]
            missing_frames.append(int(frameNo))
        np.savetxt(self.folder+"missing_frames.txt",np.array(missing_frames))
        

    def regress(self,predict=False,estimate=False,save_plot=False,beta=0.8,curve=0):
        if predict:
            import dnn
            regression = dnn.DNN()

        dataset = np.array([],dtype=np.float64).reshape(-1,4)
        total_frames = len(glob.glob(self.folder+"*kpt.txt"))        
        missing_frames = []
        if self.missing:
            missing_frames = np.int_(np.loadtxt(self.folder+"missing_frames.txt"))
            total_frames+= len(missing_frames)

        for filename in sorted(glob.glob(self.folder+"pose*")):
            try:
                # filename = self.folder+"pose"+str(x)+".txt"
                basename = os.path.basename(filename)
                frameNo = int(basename[4:-4])
                pose = np.loadtxt(filename,dtype=np.float64)
            except:
                continue
            
            data = np.append(np.array([frameNo]),pose)
            dataset = np.vstack([dataset, data])
            
        ut = util.Util()
        temp = np.copy(dataset)
        # temp = ut.weightedAvergage(temp,total_frames,beta=beta)
        
        if predict:
            if self.missing:
                regressed_pose = regression.regress(temp,missing_frames,smooth=True)
                pass
        else:
            if estimate:
                regressed_pose,gaps = ut.estimateFrames(temp, missing_frames,curve=curve)
                try:
                    gaps[1]
                    multi = True
                except:
                    multi = False
                
            else:
                regressed_pose = dataset ## Change this later

        regressed_pose = ut.weightedAvergage(regressed_pose,total_frames,beta=beta)
                 
        if save_plot:
            try:
                os.mkdir(self.folder+"plots")
            except:
                pass
            
        if estimate:            
            dataset_regressed = regressed_pose[np.int_(regressed_pose[:,0]) == 0].reshape(-1,4)
            dataset_raw = dataset[np.int_(dataset[:,0]) == 0].reshape(-1,4)                
            
            highlight = False
            
            gaps = gaps[0]
            try:
                if len(gaps[1]) > 1:
                    multi=True     
            except:
                pass
                
            pointer=0
            predicted = np.zeros_like(gaps)
            for x in range(1,len(regressed_pose)):
                data_regressed = regressed_pose[np.int_(regressed_pose[:,0]) == x].reshape(-1,4)
                data_raw = dataset[np.int_(dataset[:,0]) == x].reshape(-1,4) 
                
                if not len(data_regressed): ## Check for empty list (failed case: file does not exist)
                    continue
                
                dataset_regressed = np.vstack([dataset_regressed,data_regressed])  
                dataset_raw = np.vstack([dataset_raw,data_raw])
                
                if save_plot:
                    fig = plt.figure(figsize=(16,9))
                    ax = fig.add_subplot(111)
                    
                    if multi:                        
                        for gap in range(len(gaps)):                        
                            if x in gaps[gap]: ## Will go here if x is regressed or estimated
                                if gap == pointer:
                                    predicted[gap] = dataset_regressed[-1,:]
                                    pointer+=1
                                highlight = True
                                predicted[gap] = np.vstack([predicted[gap],data_regressed])
                            
                    else:
                        if x in missing_frames: ## Will go here if there x is predicted or regressed
                            if not highlight:
                                predicted = dataset_regressed[-1,:]
                            highlight = True
                            predicted = np.vstack([predicted,data_regressed])

    
                    ax.scatter(dataset_raw[:,0],dataset_raw[:,1],color="k",label="roll (raw)")
                    ax.scatter(dataset_raw[:,0],dataset_raw[:,2],color="b", label = "pitch (raw)")
                    ax.scatter(dataset_raw[:,0],dataset_raw[:,3],color="r", label = "yaw (raw)")
    
                    legend1 = plt.legend(loc = "upper left",fontsize="20")
                    ax.add_artist(legend1)  
                    
                    ax.plot(dataset_regressed[:,0],dataset_regressed[:,1],color="k",linewidth=3, label="roll (smooth)")
                    ax.plot(dataset_regressed[:,0],dataset_regressed[:,2],color="b",linewidth=3, label = "pitch (smooth)")
                    ax.plot(dataset_regressed[:,0],dataset_regressed[:,3],color="r",linewidth=3, label = "yaw (smooth)")
                    
                    lines = ax.get_lines()
                    
                    legend2 = plt.legend([lines[i] for i in [0,1,2]],["roll (smooth)","pitch (smooth)","yaw (smooth)"],loc = "lower right",fontsize="20")
                    ax.add_artist(legend2)
         
                    
                    if highlight:
                        if predict:
                            text = " (regressed)"
                        elif estimate:
                            text = " (estimated)"
                        if multi:
                            for gap in range(pointer):
    
                                ax.plot(predicted[gap][:,0],predicted[gap][:,1],color="g",linewidth=5, label="roll"+text)
                                ax.plot(predicted[gap][:,0],predicted[gap][:,2],color="g",linewidth=5, label = "pitch"+text)
                                ax.plot(predicted[gap][:,0],predicted[gap][:,3],color="g",linewidth=5, label = "yaw"+text)
             
                        else:
                            ax.plot(predicted[:,0],predicted[:,1],color="g",linewidth=5, label="roll"+text)
                            ax.plot(predicted[:,0],predicted[:,2],color="g",linewidth=5, label = "pitch"+text)
                            ax.plot(predicted[:,0],predicted[:,3],color="g",linewidth=5, label = "yaw"+text)
                            
                        lines = ax.get_lines()
                        legend3 = ax.legend([lines[i] for i in [3,4,5]],["roll"+text,"pitch"+text,"yaw"+text ],loc = "upper right",fontsize="20")
                        ax.add_artist(legend3)     
    
                    
                    plt.grid()
                    plt.xlim([0,total_frames])
                    plt.ylim([-90,90])
                    plt.xlabel("frames")
                    plt.ylabel("degrees")
                    plt.savefig(self.folder+"/plots/regressed"+str(int(data_regressed[0][0])).zfill(5)+".png",bbox_inches="tight")
                np.savetxt(self.folder+"regressed"+str(int(data_regressed[0][0])).zfill(5)+".txt",data_regressed[0][1:])
