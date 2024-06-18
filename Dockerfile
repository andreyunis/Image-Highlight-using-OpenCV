# Dockerfile, Image, Container
FROM python:3.9

ADD image_recognition_4.py .

RUN pip install numpy opencv-python

CMD [ "python", "./image_recognition_4.py" ]