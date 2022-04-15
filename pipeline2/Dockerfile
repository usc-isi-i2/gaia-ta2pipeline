FROM ubuntu:bionic

RUN apt-get upgrade && apt-get -y update && apt-get -y install \
    apt-utils locales unzip nano tree \
    software-properties-common wget jq vim \
    zip gcc python python-dev default-jre
#RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - && apt-get install -yq nodejs build-essential
#RUN npm install -g npm && npm install --global graphy

### Python 3.8
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh && \
    chmod +x Miniconda3-py38_4.11.0-Linux-x86_64.sh && \
    ./Miniconda3-py38_4.11.0-Linux-x86_64.sh -p /aida/miniconda -b && \
    rm Miniconda3-py38_4.11.0-Linux-x86_64.sh
ENV PATH=/aida/miniconda/bin:${PATH}
RUN conda update -y conda
# RUN pip install --upgrade pip

### Clone codebase
WORKDIR /aida/ta2-pipeline/
COPY . .
RUN chmod u+x ./entrypoint.sh

### Dependencies
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen
RUN locale-gen

RUN pip install -r requirements.txt

### Entrypoint
#CMD [ "/bin/bash", "" ]
ENTRYPOINT ["/bin/bash", "/aida/ta2-pipeline/entrypoint.sh"]
