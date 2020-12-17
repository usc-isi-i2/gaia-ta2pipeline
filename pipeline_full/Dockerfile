FROM ubuntu:16.04

RUN apt-get upgrade && apt-get -y update && apt-get -y install \
    apt-utils locales unzip nano tree \
    software-properties-common wget jq vim \
    zip gcc python python-dev default-jre
#RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - && apt-get install -yq nodejs build-essential
#RUN npm install -g npm && npm install --global graphy
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -p /aida/miniconda -b && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/aida/miniconda/bin:${PATH}
RUN conda update -y conda
# RUN pip install --upgrade pip

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen
RUN locale-gen

# cache requirements
COPY gaia-ta2pipeline/pipeline/requirements.txt /aida/ta2-pipeline/
COPY uke-coref/dependencies/aida-ukes/requirements.txt /aida/uke-coref/dependencies/aida-ukes/requirements.txt
RUN pip install -r /aida/ta2-pipeline/requirements.txt
RUN pip install -r /aida/uke-coref/dependencies/aida-ukes/requirements.txt

# named entity pipeline
WORKDIR /aida/ta2-pipeline/
COPY gaia-ta2pipeline/pipeline .
RUN chmod u+x ./entrypoint.sh
#RUN pip install -r requirements.txt

# UKE pipeline
WORKDIR /aida/uke-coref/
COPY uke-coref .

#RUN pip install -r ./dependencies/aida-ukes/requirements.txt
RUN pip install -e ./dependencies/aida-ukes
RUN pip install ./dependencies/aida-viz
RUN pip install ./dependencies/AIDA-Interchange-Format/python
RUN pip install -e .

WORKDIR /aida/ta2-full
COPY gaia-ta2pipeline/pipeline_full .
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]