FROM nginx:latest

ADD website/. /var/www/html/.
ADD website/nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx","-g","daemon off;"]
