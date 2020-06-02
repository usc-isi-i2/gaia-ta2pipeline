FROM python:3.6.3
#FROM openjdk:8
#Why is tango-common prompting?#RUN apt-get -y update
#FROM ontotext/graphdb:9.1.1-se
# docker run -p 8200:7200 --name graphdb-test -t ontotext/graphdb:9.1.1-se

LABEL maintainer=dan.napierski@toptal.com

RUN apt-get upgrade && apt-get -y update && apt-get -y install apt-utils && apt-get -y install unzip nano tree software-properties-common jq vim
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 0xB1998361219BD9C9 && apt-add-repository 'deb http://repos.azulsystems.com/ubuntu stable main' && apt-get update && apt-get -y install zulu-8
RUN pip install --upgrade pip

WORKDIR /graphdb/
COPY ./.bigfiles/graphdb-free-9.1.1-dist.zip .
RUN unzip -qq graphdb-free-9.1.1-dist.zip
ENV graphdb_home=/graphdb/graphdb-free-9.1.1
ENV PATH=${PATH}:${graphdb_home}/bin
COPY ./env.sh .
RUN chmod u+x ./env.sh

WORKDIR /maven/
COPY ./.bigfiles/apache-maven-3.6.3-bin.tar.gz .
RUN tar -xvf apache-maven-3.6.3-bin.tar.gz
ENV M2_HOME='/maven/apache-maven-3.6.3'
ENV PATH=${PATH}:${M2_HOME}/bin

WORKDIR /aida/nlp-util/
# TODO: in ./.private first do this: git clone -b graphdb-emergency-eval-fix git@github.com:usc-isi-i2/nlp-util.git
COPY ./.private/nlp-util/ .
RUN java -version
RUN mvn -version
RUN mvn install

WORKDIR /aida/ta2-pipeline/
COPY . .
RUN chmod u+x ./entrypoint.sh

RUN pip install ipykernel
RUN python -m ipykernel install --user
RUN pip install -r requirements.txt

ENV LC_ALL=en_US.UTF-8
RUN graphdb -d -s

CMD [ "/bin/bash", "" ]
