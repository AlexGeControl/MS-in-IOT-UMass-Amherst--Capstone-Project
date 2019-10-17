# base image: python openpose with GPU
FROM cwaffles/openpose:latest

# set up PYTHONPATH
ENV PYTHONPATH /openpose/build/python

# set up apt-fast:
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get -y install software-properties-common && \ 
    add-apt-repository ppa:apt-fast/stable && apt-get update && apt-get -y install apt-fast

# install packages:
RUN apt-fast update && \
    apt-fast install -y \ 
    git wget curl rsync netcat mg vim bzip2 zip unzip \
    build-essential cmake pkg-config \ 
    libjpeg8-dev libtiff5-dev \ 
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \ 
    libgtk2.0-dev \ 
    libatlas-base-dev gfortran \  
    python3-setuptools python3-tk \
    swig \ 
    libx11-6 libxcb1 libxau6 \
    xvfb dbus-x11 x11-utils \
    lxde tightvncserver \
    xfonts-base xfonts-75dpi xfonts-100dpi \
    python-pip python-dev python-qt4 \
    libssl-dev && \
    # clean up:
    apt-get autoclean -y && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install pre-requisites:
RUN pip3 install tensorflow-gpu==1.14 slidingwindow Cython

# install tf-pose-estimation:
WORKDIR /opt/
RUN git clone https://www.github.com/ildoonet/tf-pose-estimation
RUN cd tf-pose-estimation && \
    # install dependencies: 
    pip3 install -r requirements.txt && \
    # install post processing:
    cd tf_pose/pafprocess && \
    swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace && \
    # install package:
    cd /opt/tf-pose-estimation python3 setup.py install

# config vnc:
WORKDIR /root/
RUN mkdir -p /root/.vnc
COPY vnc-config/xstartup /root/.vnc/
RUN chmod a+x /root/.vnc/xstartup && \ 
    touch /root/.vnc/passwd && \ 
    /bin/bash -c "echo -e 'password\npassword\nn' | vncpasswd" > /root/.vnc/passwd && \ 
    chmod 400 /root/.vnc/passwd && \ 
    chmod go-rwx /root/.vnc && \ 
    touch /root/.Xauthority

COPY vnc-config/start-vncserver.sh /root/
RUN chmod a+x /root/start-vncserver.sh

RUN echo "mycontainer" > /etc/hostname && \ 
    echo "127.0.0.1	localhost" > /etc/hosts && \ 
    echo "127.0.0.1	mycontainer" >> /etc/hosts

# enable vnc:
EXPOSE 5901
ENV USER root

# set up workspace:
WORKDIR /workspace/
ENV PYTHONPATH $PYTHONPATH:/opt/tf-pose-estimation 

CMD [ "/root/start-vncserver.sh" ]