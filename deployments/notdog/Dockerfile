FROM tensorflow/tensorflow:latest-gpu
RUN pip install uwsgi
RUN apt-get update
RUN apt-get install 'ffmpeg'\
        'libsm6'\
        'libxext6' -y
RUN apt-get install vim -y
RUN apt-get install wget -y
RUN apt-get install nginx -y

ADD notdog/ /notdog
WORKDIR /notdog
ADD notdog/nginx.conf /etc/nginx/nginx.conf
EXPOSE 8081
RUN pip install -r requirements.txt

CMD python ./views.py

