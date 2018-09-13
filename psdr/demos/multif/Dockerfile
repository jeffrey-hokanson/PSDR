#Use the ubuntu to bootstrap the image

FROM ubuntu:latest

# Install background material
RUN apt -y update && \
	apt -y upgrade && \
	apt -y install \
		automake \
		bison \
		build-essential \ 
		cmake \
		cmake-curses-gui \
		flex \
		gfortran \
		git \
		libarpack2-dev \
		libatlas-dev \
		libfreeimage-dev \
		libfreetype6-dev \
		libgl1-mesa-dev \
		libgl2ps-dev \
		libglu1-mesa-dev \
		liblapack-pic \
		libmetis-dev \
		libopenmpi-dev \
		libsuitesparse-dev \
		libtbb-dev \
		libxi-dev \
		libxt-dev \
		libxmu-dev \
		libxmu-headers \
		mercurial \
		python-dev \
		python-matplotlib \
		python-numpy \
		python-scipy \
		python-tk \
		swig \
		tcl \
		tcl-dev \
		tcllib \
		tk \
		tklib \
		tk-dev \
		vim \
		wget 
	
# Should add apt-get clean
	
# liboce-ocaf-dev # prebuilt open cascade 

# Download all the sources here
RUN mkdir -p /opt/local/build 

# Install OpenCascade
ARG OC_VER=7.2.0
COPY opencascade-${OC_VER}.tar /opt/local/build/

RUN mkdir -p /opt/local/build/opencascade && \
	cd /opt/local/build && \
	tar xf opencascade-${OC_VER}.tar && \
	cd /opt/local/build/opencascade && \
	cmake -DCMAKE_BUILD_TYPE=RELEASE \
		-DCMAKE_CXX_FLAGS:STRING="-std=c++14" \
		-D3RDPARTY_TCL_INCLUDE_DIR=/usr/include/tcl \
		-D3RDPARTY_TK_INCLUDE_DIR=/usr/include/tk \
		/opt/local/build/opencascade-${OC_VER} \
		&& \
	make -j 4 && \
	make install && \
	cd  && \
	rm -r /opt/local/build/opencascade*  


# Open MPI (required for AEROS)
ARG OMPI_VER=3.0.0 
RUN mkdir -p /opt/local/build && \
	cd /opt/local/build && \
	wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-${OMPI_VER}.tar.bz2  && \
	tar jxvf openmpi-${OMPI_VER}.tar.bz2 && \
	cd openmpi-${OMPI_VER} && \
	./configure --prefix=/usr/local && \
	make -j4 && make install && ldconfig && \
	cd && \
	rm -r /opt/local/build 


# Install a more recent Eigen (required for AEROS)
ARG EIGEN_VER=3.3.4 
RUN mkdir -p /opt/local/build/ && \
	cd /opt/local/build && \
	wget http://bitbucket.org/eigen/eigen/get/${EIGEN_VER}.tar.bz2 && \
	tar jxvf ${EIGEN_VER}.tar.bz2 && \
	mkdir -p eigen-build && \
	cd eigen-build && \
	cmake ../eigen-eigen* && \
	make install && \
	cd  && \
	rm -r /opt/local/build 

# Install Gmsh
ARG GMSH_VER=3.0.6
RUN mkdir -p /opt/local/build && \
	cd /opt/local/build && \
	wget http://gmsh.info/src/gmsh-${GMSH_VER}-source.tgz && \
	tar xfvz gmsh-${GMSH_VER}-source.tgz && \ 
	mkdir -p /opt/local/build/gmsh && \
	cd /opt/local/build/gmsh && \
	cmake -DDEFAULT=0 \ 
		-DENABLE_FLTK:BOOL=OFF \
		-DENABLE_ANN:BOOL=ON \
		-DENABLE_MESH:BOOL=ON \
		-DENABLE_PARSER:BOOL=ON \
		-DENABLE_OCC:BOOL=ON \
		-DENABLE_BUILD_LIB=1 \
		-DENABLE_BUILD_SHARED=1 \
		-DCMAKE_CXX_FLAGS:STRING="--std=c++11" \
		-DCMAKE_INSTALL_PREFIX="/usr/local" \
		-DCMAKE_LIBRARY_PATH:PATH="/usr/local/lib" \
		/opt/local/build/gmsh-${GMSH_VER}-source && \
	make lib  && \
	make shared && \
	make install && \
	cd 
	#cd && rm -r /opt/local/build/gmsh

# Download and install SU2
# From branch nozzlethrust
ARG SU2_VER=d7a5e54f8c1033846099b79c44fa5e66454e2877
RUN mkdir -p /opt/local/build && \
	cd /opt/local/build && \
	git clone -v --progress -b darpa https://github.com/vmenier/SU2.git && \
	cd SU2 && \ 
	git checkout $SU2_VER && \
	./preconfigure.py --prefix=/usr/local/SU2 \
		--enable-mpi \
		--enable-autodiff \
		--with-cc=/usr/bin/mpicc \
		--with-cxx=/usr/bin/mpiCC && \
	make && make install && \
	cd && rm -r /opt/local/build/SU2


# Install AERO-S
#ARG AEROS_VER=ee5e20dbb527362563e3b656773077c2f0c56607
ARG AEROS_VER=cd773a5a236370c8cc779e14c9d8619ebb4ee3fc
RUN mkdir -p /opt/local/build && \
	cd /opt/local/build && \
	hg clone -v https://bitbucket.org/frg/aero-s -r ${AEROS_VER} && \
	mkdir -p /opt/local/build/aeros_build && \
	cd /opt/local/build/aeros_build && \
	cmake -DAERO=1 \
		-DCMAKE_PREFIX_PATH:PATH="/usr/local" \
		-DCMAKE_INSTALL_PREFIX=/usr/local \
		../aero-s && \
	make && make install && \
	cd / && rm -r /opt/local/build/aeros_build

# Setup the enviornment for running MULTIF
# Last command ensures that python knows where to look for gmsh
RUN echo "export SU2_RUN=/usr/local/SU2/bin" >> /etc/bash.bashrc && \ 
	echo "export SU2_HOME=/opt/local/build/SU2/" >> /etc/bash.bashrc && \
	echo "export MULTIF=/opt/local/MULTIF/" >> /etc/bash.bashrc && \ 
	echo "export PYTHONPATH=\$PYTHONPATH:\$SU2_RUN:\$MULTIF" >> /etc/bash.bashrc && \ 
	echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib" >> /etc/bash.bashrc && \ 
	echo "export PATH=\$PATH:\$SU2_RUN:/usr/local/bin" >> /etc/bash.bashrc  && \
	echo "export PS1='\h (Singularity):\w '"


# Install MULTIF
# ARG MULTIF_VER=develop
ARG MULTIF_COMMIT=01df28006fb60cd69bec620b632d493ed489b07c
RUN cd /opt/local/ && git clone -v  --progress https://github.com/vmenier/MULTIF.git && \
	cd /opt/local/MULTIF/multif && \
	git checkout $MULTIF_COMMIT && \	
	python setup.py build_ext --inplace \
		--include-dirs=/usr/local/include/gmsh:/opt/local/build/gmsh-${GMSH_VER}-source/:/usr/local/include/opencascade --library-dirs=/usr/local/lib
