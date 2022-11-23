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
		mpich \
		# MPICH provides mpicxx, but not the library required to compile against
		# which is included in libmpich-dev 
		libmpich-dev \
		&& \
	python3 -m pip install tqdm argparse

ARG SU2_VER=v6.2.0

# Install SU2
# Note that SU2 wants to keep its source around for some reason
RUN	mkdir -p ~/Source && \
	cd ~/Source && \
	git clone https://github.com/su2code/SU2.git --progress --branch ${SU2_VER} && \
	cd SU2 && \
#	git checkout ${SU2_VER}  && \
	# Build options
	export CXXFLAGS="-O3" && \
	ln -s /usr/bin/python3 /usr/bin/python && \
	# SU2 wants to install its own version of the autotools
	PATH=/root/Source/SU2/externals/autotools/bin:$PATH && \
	./preconfigure.py --enable-mpi --enable-autodiff --prefix=/usr/local/ && \ 
	make -j16 && \ 
	make install && \
	make clean && \
    echo "export SU2_RUN=/usr/local/bin/" >> /etc/bash.bashrc && \ 
	echo "export SU2_HOME=/root/Source/SU2/" >> /etc/bash.bashrc && \
	echo "export PYTHONPATH=\$SU2_RUN:\$PYTHONPATH" >> /etc/bash.bashrc



WORKDIR /workdir
ENV PYTHONPATH="/usr/local/bin"
ENV SU2_RUN="/usr/local/bin"
ENV SU2_HOME="/root/Source/SU2"
COPY naca0012_driver.py /root/naca0012_driver.py
ENTRYPOINT ["/usr/bin/python3","/root/naca0012_driver.py"]


 
