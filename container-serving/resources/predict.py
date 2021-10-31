import glob
import os
import sys
import time

sys.path.append("/fairmot/src")

import _init_paths
from tracker.multitracker import JDETracker
from tracker.basetrack import BaseTrack
from config import Config
import cv2

class FairMOTService:
    trakcer = None
    
    # class method to load trained model and create an offline predictor
    @classmethod
    def create_tracker(cls, frame_w=1920, frame_h=1080, batch_size=1):
        """load trained model"""
        try:
            model_dir = os.environ["SM_MODEL_DIR"]
        except KeyError:
            model_dir = "/opt/ml/model"

        # file path to previoulsy trained mask r-cnn model
        latest_trained_model = ""
            
        latest_trained_model = os.path.join(model_dir, "model_last.pth")
        if not os.path.exists(latest_trained_model):
            model_search_path = os.path.join(model_dir, "model_*.pth")
            for model_file in glob.glob(model_search_path):
                if model_file > latest_trained_model:
                    latest_trained_model = model_file
        
        print(f"Using model: {latest_trained_model}")
            
        # create a config for FairMOT model
        config = Config(load_model=latest_trained_model, frame_rate=25)
        config.frame_w = frame_w
        config.frame_h = frame_h

        cls.trakcer = JDETracker(config, batch_size=batch_size)
        return cls.trakcer
        
    @classmethod
    def get_tracker(cls):
        return cls.trakcer

import base64
import json

from flask import Flask, Response, request

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def health_check():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully and crrate a predictor."""
    health = True
    
    if not FairMOTService.get_tracker():
        health = FairMOTService.create_tracker() is not None  # You can insert a health check here
    status = 200 if health else 404
    return Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def inference():
    if not request.is_json:
        result = {"error": "Content type is not application/json"}
        return Response(response=result, status=415, mimetype="application/json")

    path = None
    try:
        content = request.get_json()
        frame_id = content["frame_id"]
        
        print(f'Received Frame-{frame_id}')
        if int(frame_id) == 0:
            frame_w = content["frame_w"]
            frame_h = content["frame_h"]
            batch_size = content["batch_size"]
            
            tracker = FairMOTService.create_tracker(frame_w, frame_h, batch_size)
        else:
            tracker = FairMOTService.get_tracker()
            
        path = os.path.join("/tmp", str(frame_id))
        with open(path, "wb") as fh:
            s = time.time()
            ing_data_string = content["frame_data"]
            img_data_bytes = bytearray(ing_data_string, encoding="utf-8")
            fh.write(base64.decodebytes(img_data_bytes))
            fh.close()
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            online_targets = tracker.update([img])
            print(f'Processing time: {time.time()-s}')

            return Response(response=json.dumps(online_targets), status=200, mimetype="application/json")
    except Exception as e:
        print(str(e))
        result = {"error": f"Internal server error"}
        return Response(response=result, status=500, mimetype="application/json")
    finally:
        if path:
            os.remove(path)
