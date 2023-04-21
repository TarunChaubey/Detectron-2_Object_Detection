from flask import Flask, request,jsonify,render_template
from src.utils import PredicImage
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import os

import numpy as np


import logging

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


os.makedirs('logs',exist_ok=True)
logging.basicConfig(filename='logs/logfile.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

logging.info('API Statred')
logging.info('module import successfully')

upload_dir = "./uploadeImages/"
os.makedirs(upload_dir,exist_ok=True)
logging.info('now working on flask app')


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("./COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
# cfg.MODEL.WEIGHTS = "./OutputModels/model_final.pth"
cfg.MODEL.WEIGHTS = "./OutputModelsDect/model_final.pth"
cfg.MODEL.DEVICE = "cpu"



classes = ["benign","malignant"]
dataset_name = "val"
MetadataCatalog.get(dataset_name).set(things_classes = classes)
register_coco_instances("val",{},"Data/val.json","Data/val")
test_metadata = MetadataCatalog.get('val')

predictor = DefaultPredictor(cfg)

app = Flask(__name__)

CORS(app)

logging.info('index url called')
@app.route("/")
def index():
  return render_template('upload.html')


logging.info('predict url called')
@app.route("/predict",methods=['POST'])
def predict():
  file = request.files['image']
  file.save(upload_dir+file.filename)

  logging.info(f"image upload at {upload_dir+file.filename}")
  img = Image.open(upload_dir+file.filename)
  imgarray = np.array(img)[:, :, ::-1]
  res = PredicImage(imgarray)

  logging.info(f"working to predict {file.filename}")
  pred_classes = res["instances"].to("cpu").pred_classes.tolist()[0]
  pred_scores = res["instances"].to("cpu").scores.tolist()[0]

  # Convert the predicted class labels and scores to a dictionary
  output = {
      "Actual File":file.filename,
      "Predicted File":classes[pred_classes],
      "Predicted": pred_classes,
      "scores": pred_scores

  }

  logging.info(f'prediction result return {output}')
  return jsonify(output)

if __name__ == "__main__":
  app.run()