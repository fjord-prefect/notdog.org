FROM tensorflow/tensorflow:latest-gpu
RUN pip install uwsgi
RUN apt-get update
RUN apt-get install 'ffmpeg'\
	'libsm6'\
	'libxext6' -y
RUN apt-get install vim -y
RUN apt-get install wget -y
RUN apt-get install nginx -y

ADD breed/ /breed
WORKDIR /breed
ADD breed/nginx.conf /etc/nginx/nginx.conf
EXPOSE 8181 
RUN pip install -r requirements.txt

CMD python ./views.py
