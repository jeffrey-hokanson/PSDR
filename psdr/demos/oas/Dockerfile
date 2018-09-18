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

RUN mkdir -p ~/Source && \
	cd ~/Source && \
	git clone https://github.com/OpenMDAO/OpenMDAO1.git && \
	cd OpenMDAO1 && \
	git checkout ${OM_VER} && \
	cd .. && pip install -e OpenMDAO1

RUN cd ~/Source && \
	git clone https://github.com/jeffrey-hokanson/OpenAeroStruct && \
	cd OpenAeroStruct &&  \
	make 

COPY oas_driver.py /root/Source/oas_driver.py

WORKDIR /root/tmp
ENTRYPOINT ["python","/root/Source/oas_driver.py"]
 
