#Use the ubuntu to bootstrap the image

FROM ubuntu:disco


# Install background material
RUN apt-get -y update && \
	apt-get -y upgrade && \
	apt-get -y install --no-install-recommends \
		build-essential \ 
		gfortran \
		openssl \
		git \
		python3 \
		python3-pip \
		python3-dev \
		python3-numpy \
		python3-pip \
		python3-scipy \
#		vim \
		debianutils \
		m4 \
		perl \
		autoconf \
		automake && \
	python3 -m pip install tqdm

ARG SU2_VER=v6.2.0

# Install SU2
RUN	mkdir -p ~/Source && \
	cd ~/Source && \
	git clone https://github.com/su2code/SU2.git && \
	cd SU2 && \
	git checkout ${SU2_VER}  && \
	./bootstrap && \
	./configure && \
	make -j5 && \ 
	make install && \
	cd /root/ && \
	rm -r /root/Source/SU2


COPY su2_driver.py /root/Source/su2_driver.py
COPY mesh_out.su2 /root/Source/mesh_out.su2
COPY def_NACA0012.cfg /root/Source/def_NACA0012.cfg
COPY inv_NACA0012_def.cfg /root/Source/inv_NACA0012_def.cfg

WORKDIR /root/tmp
ENTRYPOINT ["/usr/bin/python3","-u","/root/Source/su2_driver.py"]


 
