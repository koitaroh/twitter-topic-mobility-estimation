FROM python:3.5.2

MAINTAINER Satoshi Miyazawa <koitaroh@gmail.com>

RUN mkdir -p /src
WORKDIR /src
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DOCKER true

RUN apt-get update
RUN apt-get install -y mysql-client build-essential gfortran gcc g++ mysql-client emacs git
RUN apt-get install mecab libmecab-dev mecab-ipadic mecab-ipadic-utf8 python-mecab
RUN apt-get install build-essential python3-dev python3-setuptools python3-pip python3-numpy python3-scipy python3-matplotlib python3-pandas python3-venv


RUN sudo cp -p /usr/share/zoneinfo/Japan /etc/localtime


ADD ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

