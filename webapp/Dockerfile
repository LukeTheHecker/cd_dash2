#Grab the latest alpine image
# This works:
# FROM python:3


# FROM alpine:3.10
# FROM python:3.7.7-alpine
FROM continuumio/miniconda3:latest

# RUN apk add --no-cache --update python3 py3-pip bash

# RUN apk add --update-cache python3 py-pip

COPY requirements.txt /
# Copy requirements.txt
# ADD ./webapp/requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install --upgrade pip
RUN conda install python=3.7
RUN pip install numpy

RUN pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl
# RUN pip install tflite-runtime==2.1.0.post1
# RUN pip3 install numpy
RUN pip install -r requirements.txt

# Add our code
RUN mkdir /myworkdir
WORKDIR /myworkdir
COPY ./ ./

CMD ["python", "index.py"]
# CMD gunicorn --bind 0.0.0.0:$PORT wsgi 