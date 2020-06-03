import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


class Video():
    def __init__(self,img_path, annot_path,img_format="jpeg",plot=False,trans=False,fps=20):
        if plot:
            plot_path = annot_path+"plots/"
        headpose = annot_path.split("/")[-2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = fps
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        num_images = len(glob.glob(img_path+"*"+img_format))
        for k in range(num_images):
            try:
                img = cv2.imread(img_path+"frame"+str(k)+"."+img_format)
                len(img)
            except:
                continue

            try:
                
                kp = glob.glob(annot_path+"frame"+str(k)+"_*"+"kpt.txt")[0]
                top,left = int(kp.split("_")[1]), int(kp.split("_")[2])    
                
                pose = np.loadtxt(annot_path+"pose"+str(k)+".txt") 
                
                roll = "roll: "+str(-pose[0])[:7] + " (raw)"
                pitch = "pitch: "+str(pose[1])[:7]+ " (raw)"
                yaw = "yaw: "+str(pose[2])[:7]+ " (raw)"

                keypoints = np.loadtxt(kp)
                for (x,y,z) in keypoints:
                    cv2.circle(img,(int(x)+top,int(y)+left), 2,(255,255,255),-1)
            except:
                pitch = "pitch: NA"
                roll = "roll: NA"
                yaw = "yaw: NA"
                
            cv2.putText(img,roll,(20,50),font,1,(0,0,0),thickness=3)
            cv2.putText(img,pitch,(20,100),font,1,(255,0,0),thickness=3)
            cv2.putText(img,yaw,(20,150),font,1,(0,0,255),thickness=3)
            

            if plot and (k!=0):
                try:
                    img_plot = cv2.imread(plot_path+"regressed"+str(k)+".png")
                    len(img_plot)
                    self.img_plot = img_plot
                except:
                    pass
                H,W = img.shape[:-1]
                h,w = self.img_plot.shape[:-1]
            else:
                video = cv2.VideoWriter("./Results/"+headpose+".mp4",fourcc,fps,(W,H))
                if trans:
                    sub_img = img[H-h:,:w,:]
                    sub_img[self.img_plot!=255] = 0
                    
                    self.img_plot[self.img_plot==255] = 0
                    add = sub_img + self.img_plot
                    
                    img[H-h:,:w,:] = add        
                else:
                    img[H-h:,:w,:] = self.img_plot
            
            video.write(img)
        video.release()