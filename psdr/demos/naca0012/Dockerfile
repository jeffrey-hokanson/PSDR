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
	python3 -m pip install tqdm argparse

ARG SU2_VER=v6.2.0

# Install SU2
# Note that SU2 wants to keep its source around for some reason
RUN	mkdir -p ~/Source && \
	cd ~/Source && \
	git clone https://github.com/su2code/SU2.git && \
	cd SU2 && \
	git checkout ${SU2_VER}  && \
	./bootstrap && \
	./configure && \
	make -j16 && \ 
	make install && \
	make clean && \
    echo "export SU2_RUN=/usr/local/bin/" >> /etc/bash.bashrc && \ 
	echo "export SU2_HOME=/root/Source/SU2/" >> /etc/bash.bashrc && \
	echo "export PYTHONPATH=\$SU2_RUN:\$PYTHONPATH" >> /etc/bash.bashrc

COPY naca0012_driver.py /root/naca0012_driver.py


WORKDIR /workdir
ENV PYTHONPATH="/usr/local/bin"
ENV SU2_RUN="/usr/local/bin"
ENV SU2_HOME="/root/Source/SU2"
ENTRYPOINT ["/usr/bin/python3","/root/naca0012_driver.py"]


 
