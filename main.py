import util
import os
import demo
import warnings
warnings.filterwarnings("ignore")
import argparse
import shutil
import pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", action="store", dest="t_p", type=str, help="Test path")
    parser.add_argument("-f", action="store_true", dest="f_d", default=False, help="Force detection")
    args = vars(parser.parse_args())
    test_path = args["t_p"] 
    
    if test_path == "":
        print("No test path given!")        
        return
        
    kpt_file_path = os.path.join(test_path, "kpt.txt")
    kpt_file_exists = os.path.isfile(kpt_file_path)
    force_detection = args["f_d"]
    img_format = "jpeg"
        
    ut = util.Util(test_path, roi=300, image_format=img_format)
    if not kpt_file_exists or force_detection:
        
        ut.processImage()
        
        demo.PRNet(test_path)
        
        shutil.rmtree(os.path.join(test_path, "temp"))
    
    ut.readKPT()
    head_pose = pose.Pose(test_path)
    head_pose.regress(curve=3)
    
    
    
    
if __name__ == "__main__":

    main()
