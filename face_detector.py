import cv2
import dlib
from imutils import face_utils


class FaceDetector():
    
    def dlib_frontal_face(self,image):
        x,y,w,h = -1,-1,-1,-1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_detect = dlib.get_frontal_face_detector()
        rects = face_detect(gray, 1)
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 2)
        # self.show(gray)
        return x,y,x+w,y+h
    
        
    def dlib_cnn(self,image):
        x1,y1,x2,y2 = -1,-1,-1,-1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dnnFaceDetector = dlib.cnn_face_detection_model_v1("./detection_models/mmod_human_face_detector.dat")
        rects = dnnFaceDetector(gray, 1)
        for (i, rect) in enumerate(rects):
            x1 = rect.rect.left()
            y1 = rect.rect.top()
            x2 = rect.rect.right()
            y2 = rect.rect.bottom()
        return x1,y1,x2,y2
    
        
    def opencv_cnn(self,image):
        x1,y1,x2,y2 = -1,-1,-1,-1
        frameHeight, frameWidth = image.shape[:-1]
        modelFile = "./detection_models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "./detection_models/deploy.prototxt.txt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # self.show(image)
        return x1,y1,x2,y2
    
    def detect(self,image,frameNo):
        success = True
        top,left,bottom,right = self.opencv_cnn(image)
        if top == -1 and frameNo == 0:
            top,left,bottom,right = self.dlib_frontal_face(image)
            if top == -1:
                success = False
        elif top == -1:
            success = False            
        return success,[top,left,bottom,right]
    
    def show(self,image):
        cv2.imshow("img",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        













    


