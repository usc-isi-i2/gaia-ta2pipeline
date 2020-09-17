FROM python:3.6.3

LABEL maintainer="dan.napierski@toptal.com"
LABEL name="GAIA AIDA TA2"
LABEL remarks="https://hub.docker.com/u/gaiaaida"
LABEL version=0
LABEL revision=1

RUN apt-get upgrade && apt-get -y update && apt-get -y install apt-utils locales unzip nano tree software-properties-common jq vim zip
RUN pip install --upgrade pip


WORKDIR /aida/ta2-pipeline/
COPY . .
RUN chmod u+x ./entrypoint.sh


ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8


RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen
RUN locale-gen

RUN pip install -r requirements.txt

#CMD [ "/bin/bash", "" ]
ENTRYPOINT ["/bin/bash", "/aida/ta2-pipeline/entrypoint.sh"]
