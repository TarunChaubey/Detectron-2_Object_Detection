# docker build --rm . -t ylashin/detectron2:latest
# docker run -it ylashin/detectron2:latest bin/bash
# docker run -p 8181:5000 -it ylashin/detectron2:latest bin/bash
# docker run -p 8181:5000 -d ylashin/detectron2:latest
# docker push ylashin/detectron2:latest

FROM python:3.8-slim-buster

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev

# Detectron2 prerequisites
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Detectron2 - CPU copy
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html

# Development packages
RUN pip install flask flask-cors requests opencv-python pyyaml==5.1

WORKDIR /app

COPY app.py app.py

ENTRYPOINT ["python", "/app/app.py"]