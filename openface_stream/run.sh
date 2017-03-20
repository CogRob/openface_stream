nvidia-docker run \
  -it \
  -v ~/Desktop/face_data:/root/face_data \
  openfacestream_openface_stream bash -c "src/openface_trainer.py /root/face_data"

  # openfacestream_openface_stream bash
