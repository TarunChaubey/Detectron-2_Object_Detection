# uvicorn FastAPI:app

from fastapi.responses import HTMLResponse, JSONResponse,FileResponse
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import uvicorn
import cv2
import os

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo



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

app = FastAPI()
templates = Jinja2Templates(directory="templates")
classes = ["benign","malignant"]

upload_dir = "./uploadeImages/"
predict_dir = "./APIPredictedImages/"
os.makedirs(upload_dir,exist_ok=True)
os.makedirs(predict_dir,exist_ok=True)



@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/predict")
# async def predict(image: UploadFile = File(...)):
async def predict(request: Request,image: UploadFile = File(...)):
    contents = await image.read()
    file_path = os.path.join(upload_dir, image.filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    img = cv2.imread(file_path)
    # perform prediction on the image
    # print(file_path)

    img = Image.open(file_path)
    imgarray = np.array(img)[:, :, ::-1]
    res = predictor(imgarray)

    predict_fle = os.path.join(predict_dir,image.filename)
    v = Visualizer(img, metadata=test_metadata, scale=0.8)
    v = v.draw_instance_predictions(res["instances"].to("cpu"))
    cv2.imwrite(predict_fle,v.get_image()[:, :, ::-1]) #save predicted images
    
    pred_classes = res["instances"].to("cpu").pred_classes.tolist()[0]
    pred_scores = res["instances"].to("cpu").scores.tolist()[0]
    pred_boxes = res["instances"].to("cpu").pred_boxes

    # Convert the predicted class labels and scores to a dictionary
    pred_img_path = predict_fle.replace("./","../") # only for path modification of pred_img for html page
    output = {
        "Predicted": pred_classes,
        "scores": pred_scores,
        "pred_boxes":pred_boxes

    }
    
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "pred_img_path": pred_img_path, "text": output,"filename":image.filename}
    )
    # return FileResponse(predict_fle)

if __name__ == "__main__":
    uvicorn.run(app, debug=True)