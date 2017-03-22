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
	@docker build --tag=cogrob/openface_stream:cuda openface_stream/docker
	@docker build --tag=cogrob/openface_stream openface_stream
	@docker build --tag=cogrob/openface_stream:ros ros_stream

pull:
	@docker pull cogrob/openface_stream:cuda
	@docker pull cogrob/openface_stream
	@docker pull cogrob/openface_stream:ros

push:
	@docker push cogrob/openface_stream:cuda
	@docker push cogrob/openface_stream
	@docker push cogrob/openface_stream:ros

clean:
	@docker rmi -f cogrob/openface_stream:ros
	@docker rmi -f cogrob/openface_stream
	@docker rmi -f cogrob/openface_stream:cuda
