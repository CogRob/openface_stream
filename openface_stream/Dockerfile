# FROM bamos/openface
# FROM ubuntu:14.04
FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y \
      python-argparse \
      python-imaging \
      python-opencv \
      byobu && \
    rm -rf /var/lib/apt/lists/*

ENV WORKSPACE /root/openface_stream
ADD . $WORKSPACE
WORKDIR $WORKSPACE
