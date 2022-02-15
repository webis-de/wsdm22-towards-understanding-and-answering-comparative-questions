FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

USER root
COPY . /app
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip git build-essential libssl-dev libffi-dev

RUN export PYTHONIOENCODING=utf8
RUN LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
RUN export LD_LIBRARY_PATH
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade tqdm torch torchvision

COPY . .
