#Grab the latest alpine image
FROM python:3


# Copy requirements.txt
ADD ./webapp/requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install tflite
RUN pip install -r /tmp/requirements.txt

# Add our code
ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp

			
CMD gunicorn --bind 0.0.0.0:$PORT wsgi 