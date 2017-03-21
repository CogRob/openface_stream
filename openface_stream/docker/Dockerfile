# Start with cuDNN base image
FROM nvidia/cuda:8.0-cudnn5-devel
# MAINTAINER Kai Arulkumaran <design@kaixhin.com>

# Install git, apt-add-repository and dependencies for iTorch
RUN apt-get update && apt-get install -y \
  git \
  software-properties-common \
  ipython3 \
  libssl-dev \
  libzmq3-dev \
  python-zmq \
  python-pip \
  sudo

# Install Jupyter Notebook for iTorch
RUN pip install notebook ipywidgets

# Run Torch7 installation scripts (dependencies only)
RUN git clone https://github.com/torch/distro.git /root/torch --recursive && cd /root/torch && \
  bash install-deps

# Start with CUDA Torch dependencies
# FROM kaixhin/cuda-torch-deps:8.0
# MAINTAINER Kai Arulkumaran <design@kaixhin.com>

# Run Torch7 installation scripts
RUN cd /root/torch && \
# Run without nvcc to prevent timeouts
  sed -i 's/path_to_nvcc=$(which nvcc)/path_to_nvcc=$(which no_nvcc)/g' install.sh && \
  sed -i 's,path_to_nvcc=/usr/local/cuda/bin/nvcc,path_to_nvcc=,g' install.sh && \
  ./install.sh

# Export environment variables manually
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

# Set ~/torch as working directory
WORKDIR /root/torch

# Start with CUDA Torch dependencies 2
# FROM kaixhin/cuda-torch-deps:2-8.0
# MAINTAINER Kai Arulkumaran <design@kaixhin.com>

# Restore Torch7 installation script
RUN sed -i 's/path_to_nvcc=$(which no_nvcc)/path_to_nvcc=$(which nvcc)/g' install.sh

# Install CUDA libraries
RUN luarocks install cutorch && \
  luarocks install cunn && \
  luarocks install cudnn

# FROM ubuntu:14.04
# MAINTAINER Brandon Amos <brandon.amos.cs@gmail.com>

RUN apt-get update && \
    apt-get install -y --force-yes \
    build-essential \
    cmake \
    curl \
    gfortran \
    git \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libopencv-dev \
    libswscale-dev \
    pkg-config \
    python-dev \
    python-numpy \
    python-protobuf\
    software-properties-common \
    zip \
        python-pip \
        python-nose \
        python-opencv \
        python-scipy \
        python-pandas \
        wget \
        sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN luarocks install nn && \
    luarocks install dpnn && \
    luarocks install image && \
    luarocks install optim && \
    luarocks install csvigo && \
    luarocks install torchx && \
    luarocks install tds

# RUN cd ~ && \
#     mkdir -p ocv-tmp && \
#     cd ocv-tmp && \
#     curl -L https://github.com/Itseez/opencv/archive/2.4.11.zip -o ocv.zip && \
#     unzip ocv.zip && \
#     cd opencv-2.4.11 && \
#     mkdir release && \
#     cd release && \
#     cmake -D CMAKE_BUILD_TYPE=RELEASE \
#           -D CMAKE_INSTALL_PREFIX=/usr/local \
#           -D BUILD_PYTHON_SUPPORT=ON \
#           .. && \
#     make -j8 && \
#     make install && \
#     rm -rf ~/ocv-tmp

WORKDIR /root/
RUN git clone https://github.com/davisking/dlib.git && cd dlib && \
    git checkout tags/v19.4 && \
    cd python_examples && \
    mkdir build && cd build && \
    cmake \
      -DUSE_AVX_INSTRUCTIONS=ON \
      ../../tools/python && \
    cmake \
      --build . \
      --config Release \
      -- -j $(nproc) && \
    cp dlib.so /usr/local/lib/python2.7/dist-packages && \
    rm -rf ~/dlib-tmp

WORKDIR /root
RUN git clone https://github.com/cmusatyalab/openface.git && \
    cd openface && \
    ./models/get-models.sh && \
    pip2 install -r requirements.txt && \
    python2 setup.py install && \
    pip2 install -r demos/web/requirements.txt && \
    pip2 install -r training/requirements.txt

EXPOSE 8000 9000
CMD /bin/bash -l -c '/root/openface/demos/web/start-servers.sh'
