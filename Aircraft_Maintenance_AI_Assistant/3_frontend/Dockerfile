FROM python:3.9-slim-buster
LABEL MAINTAINER="Aircraft Maintenance AI Team"

ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    vim \
    tesseract-ocr \
    libopencv-dev \
    poppler-utils \
    tzdata \
    libgl1-mesa-glx

RUN apt-get install -y python3-distutils

# RUN pip install --upgrade pip
# RUN wget https://bootstrap.pypa.io/get-pip.py
# RUN python get-pip.py
# RUN python3 -m pip install --trusted-host pypi.org boto3 awscli

# RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

RUN mkdir ~/.streamlit

COPY . /app
RUN mv /app/.streamlit ~/.streamlit
WORKDIR /app
RUN python3 -m pip install -r requirements.txt


EXPOSE 8501

HEALTHCHECK CMD curl --fail http://0.0.0.0:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Main.py", "--server.address=0.0.0.0", "&"]