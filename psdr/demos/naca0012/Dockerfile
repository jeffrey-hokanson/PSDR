#Use the ubuntu to bootstrap the image

ARG OM_VER=d71f7477a82c95860d547e40f06ecc081945c179
ARG OAS_VER=8ee82d3e6edf87b9d8435773c4b4531e1c66ca2b

FROM ubuntu:latest

# Install background material
RUN apt -y update && \
	apt -y upgrade && \
	apt -y install \
		build-essential \ 
		gfortran \
		git \
		python-dev \
		python-numpy \
		python-pip \
		python-scipy \
		vim && \
	pip install tqdm 

RUN	mkdir -p ~/Source && \
	cd ~/Source && \
	git clone https://github.com/su2code/SU2.git && \
	cd SU2 && \
	git checkout v6.1.0 && \
	./configure && \
	make && \
	make install


 
