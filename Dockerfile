#Grab the latest alpine image
FROM heroku/miniconda:3


# Copy requirements.txt
ADD ./webapp/requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-win_amd64.whl
RUN pip install -r /tmp/requirements.txt

# Add our code
ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp

			
CMD gunicorn --bind 0.0.0.0:$PORT wsgi 