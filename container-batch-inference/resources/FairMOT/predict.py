# Batch inference
import sys
import os
import logging
import glob
import subprocess

sys.path.append("/fairmot/src")

import _init_paths
from tracker.multitracker import JDETracker
from config import Config
import cv2
import numpy as np

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def draw_res(tracker_dict, frame, frame_id, image_w):
    i = 0
    indexIDs = []
    boxes = []
    person_num = 0
    conf = None
    text_scale = max(1, image_w / 1600.)
    text_thickness = 1
    line_thickness = max(1, int(image_w/ 500.))
    for track_id, tlwh in tracker_dict.items():
        indexIDs.append(track_id)
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        color = get_color(abs(int(track_id)))
        cv2.rectangle(frame, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(frame, str(track_id), (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 0),thickness=1)
        cv2.putText(frame, 'frame:{}'.format(frame_id), (int(25), int(25)),0, text_scale, (0,0,255),1)
        i += 1
    return frame

class FairMOTService:
    trakcer = None
    
    # class method to load trained model and create an offline predictor
    @classmethod
    def create_tracker(cls, frame_w=1920, frame_h=1080, batch_size=1):
        """load trained model"""
        subprocess.run("tar -xf /opt/ml/processing/model/model.tar.gz -C /opt/ml/processing/model", shell=True)
        latest_trained_model = "/opt/ml/processing/model/model_last.pth"
            
        # create a config for FairMOT model
        config = Config(load_model=latest_trained_model, frame_rate=25)
        config.frame_w = frame_w
        config.frame_h = frame_h

        cls.trakcer = JDETracker(config, batch_size=batch_size)
        return cls.trakcer

def infer_video(vpath, width, height):
    tracker = FairMOTService.create_tracker(frame_w=width, frame_h=height)
    
    cap = cv2.VideoCapture(vpath)
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_path = os.path.join('/opt/ml/processing/output', vpath.split('/')[-1])
    print(f'output_path: {output_path} width: {width}, height: {height}')
    out = cv2.VideoWriter(output_path, fourcc, 25, (width, height))
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if ret != True:
            break
        
        online_targets = tracker.update([frame])
        frame_res = draw_res(online_targets[0], frame, frame_id, width)
        out.write(frame_res)
        frame_id += 1
        print(f'frame-{frame_id}')
    
    out.release()
    cap.release()

def infer_frames():
    pass

def check_data():
    input_dir = "/opt/ml/processing/input"
    data_search_path = os.path.join(input_dir, "*.mp4")
    data_list = glob.glob(data_search_path)
    
    video_info = {}
    
    for vpath in data_list:
        print(f'Start processing {vpath}')
        cap = cv2.VideoCapture(vpath)
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        video_info[vpath]={'width': int(width), 'height': int(height)}
        print(f'Finish processing {vpath}')
    return video_info
    
def main():    
    video_info = check_data()
    for k, v in video_info.items():
        infer_video(k, v['width'], v['height'])

if __name__ == "__main__":
    main()
    