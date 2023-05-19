FROM ubuntu:22.04

COPY . ./neural-abstraction
COPY ./spaceex_exe ./neural-abstraction/spaceex_exe

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update &&\
    apt-get install -y sudo curl vim python3 python3-pip tzdata libgmp3-dev plotutils &&\
    curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install.sh | sudo bash &&\
    pip3 install -r neural-abstraction/requirements.txt &&\
    chmod +x ./neural-abstraction/install_flowstar.sh &&\
    ./neural-abstraction/install_flowstar.sh

# WORKDIR /neural-abstraction
