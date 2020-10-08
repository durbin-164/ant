# base
FROM nvidia/cuda:11.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt update -y && useradd -m docker
RUN apt install -y cmake
RUN apt install -y git nano

#Install sonar dependency
RUN apt-get install -y zip unzip
# RUN snap install cmake --classic

RUN apt-get install -y curl jq build-essential libssl-dev libffi-dev python3 python3-venv python3-dev


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

######## using for documentation and github pages#####

RUN apt install -y doxygen sphinx-doc python3-pip rsync

RUN pip3 install sphinx-rtd-theme \
    && pip3 install breathe \
    && pip3 install sphinx-sitemap

# ****************Start for github runner ********************

# RUN apt update -y && useradd -m docker

# set the github runner version
ARG RUNNER_VERSION="2.273.5"

# update the base packages and add a non-sudo user

# install python and the packages the your code depends on along with jq so we can parse JSON
# add additional packages as necessary
# RUN apt-get install -y jq build-essential libssl-dev libffi-dev python3 python3-venv python3-dev

# cd into the user directory, download and unzip the github actions runner
RUN cd /home/docker && mkdir actions-runner && cd actions-runner \
    && curl -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    && tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# install some additional dependencies
RUN chown -R docker ~docker && /home/docker/actions-runner/bin/installdependencies.sh

# copy over the start.sh script
COPY start.sh start.sh

# make the script executable
RUN chmod +x start.sh

# since the config and run script for actions are not allowed to be run by root,
# set the user to "docker" so all subsequent commands are run as the docker user
USER docker

# set the entrypoint to the start.sh script
ENTRYPOINT ["./start.sh"]