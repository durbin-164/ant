FROM nvidia/cuda:10.2-base-ubuntu18.04

#Install miniconda

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
RUN conda create -y -q -n avenv python=3.8

#Install dependency
RUN apt-get update
RUN apt-get install -y libgtest-dev
RUN apt-get install -y python3-setuptools
RUN apt-get install -y python-dev

#Install Arrayfire

RUN wget --quiet http://arrayfire.s3.amazonaws.com/3.7.2/ArrayFire-v3.7.2_Linux_x86_64.sh
RUN chmod +x ./ArrayFire-v3.7.2_Linux_x86_64.sh
RUN ./ArrayFire-v3.7.2_Linux_x86_64.sh --skip-license --exclude-subdir --prefix=/opt
RUN rm ./ArrayFire-v3.7.2_Linux_x86_64.sh
RUN sh -c "echo /opt/arrayfire/lib64 >> /etc/ld.so.conf.d/arrayfire.conf"
RUN ldconfig

#Install sonar dependency
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y zip unzip
RUN apt-get install -y build-essential cmake 


# Install Sonar-build-wrapper
RUN curl -s -L https://sonarcloud.io/static/cpp/build-wrapper-linux-x86.zip -o sonarwrapper.zip \
  && unzip -qq sonarwrapper.zip \
  && rm -rf sonarwrapper.zip \
  && mv build-wrapper-linux-x86 build-wrapper

ENV PATH $PATH:/build-wrapper/


# Install Sonar-scanner-CLI
RUN curl -s -L https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.4.0.2170-linux.zip -o sonarscanner.zip \
  && unzip -qq sonarscanner.zip \
  && rm -rf sonarscanner.zip \
  && chmod 755 sonar-scanner-4.4.0.2170-linux/jre/bin/java\
  && mv sonar-scanner-4.4.0.2170-linux sonar-scanner

ENV SONAR_RUNNER_HOME=sonar-scanner
ENV PATH $PATH:/sonar-scanner/bin