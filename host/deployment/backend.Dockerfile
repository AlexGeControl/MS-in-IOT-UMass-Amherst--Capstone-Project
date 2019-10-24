FROM tiangolo/uwsgi-nginx-flask:latest

RUN pip3 install pymongo dnspython Flask-PyMongo