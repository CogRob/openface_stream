FROM cogrob/openface_stream:cuda

RUN apt-get update && \
    apt-get install -y \
      byobu && \
    rm -rf /var/lib/apt/lists/*

ENV WORKSPACE /root/openface_stream
ADD . $WORKSPACE
WORKDIR $WORKSPACE
