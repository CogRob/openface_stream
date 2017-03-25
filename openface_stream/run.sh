nvidia-docker run \
  -it \
  -v ~/Desktop/ftp/Check-In:/root/data \
  openfacestream_openface_stream bash -c "src/openface_trainer.py --input /root/data"

  # openfacestream_openface_stream bash
