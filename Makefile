all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build  - build images"
	@echo "   1. make pull   - pull images"
	@echo "   1. make clean  - remove images"
	@echo ""

build:
	@docker build --tag=cogrob/openface_stream openface_stream
	@docker build --tag=cogrob/ros_stream ros_stream

pull:
	@docker pull cogrob/openface_stream
	@docker pull cogrob/ros_stream

clean:
	@docker rmi -f cogrob/openface_stream
	@docker rmi -f cogrob/ros_stream
