FROM nginx:latest
COPY nginx.conf /etc/nginx/nginx.conf
# COPY logrotate.conf /etc/logrotate.conf

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]