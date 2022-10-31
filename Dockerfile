FROM python:3
ADD requirements.txt /
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
ADD src/*.* /src/
ADD weights/* /weights
CMD [ "python", ".src/main.py" ]