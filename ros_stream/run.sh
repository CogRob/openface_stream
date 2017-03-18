docker run \
  -it \
  --publish 8080:8080 \
  --device /dev/video0:/dev/video0 \
  cogrob/ros_stream roslaunch bringup.launch
