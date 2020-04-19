FROM python:3

COPY requirements.txt /
RUN pip install -r /requirements.txt

RUN mkdir /myworkdir
WORKDIR /myworkdir
COPY ./ ./

EXPOSE 8050
CMD ["python", "./app-simple.py"]