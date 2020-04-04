FROM python:3.6.3
#FROM openjdk:8
#Why is tango-common prompting?#RUN apt-get -y update
#FROM ontotext/graphdb:9.1.1-se
# docker run -p 8200:7200 --name graphdb-test -t ontotext/graphdb:9.1.1-se

LABEL maintainer=dan.napierski@toptal.com

RUN apt-get upgrade && apt-get -y update && apt-get -y install apt-utils && apt-get -y install unzip nano tree software-properties-common 
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 0xB1998361219BD9C9 && apt-add-repository 'deb http://repos.azulsystems.com/ubuntu stable main' && apt-get update && apt-get -y install zulu-8
RUN pip install --upgrade pip

WORKDIR /graphdb/
COPY ./.bigfiles/graphdb-free-9.1.1-dist.zip .
RUN unzip -qq graphdb-free-9.1.1-dist.zip
ENV graphdb_home=/graphdb/graphdb-free-9.1.1
ENV PATH=${PATH}:${graphdb_home}/bin

WORKDIR /aida/ta2-pipeline/
COPY . .

RUN pip install -r requirements.txt

RUN graphdb -d -s

CMD [ "/bin/bash", "" ]
