FROM python:3.6.3

LABEL maintainer="dan.napierski@toptal.com"
LABEL name="GAIA AIDA TA2"
LABEL remarks="https://hub.docker.com/u/gaiaaida"
LABEL version=0
LABEL revision=1

RUN apt-get upgrade && apt-get -y update && apt-get -y install apt-utils locales unzip nano tree software-properties-common jq vim
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 0xB1998361219BD9C9 && apt-add-repository 'deb http://repos.azulsystems.com/ubuntu stable main' && apt-get update && apt-get -y install zulu-8
RUN pip install --upgrade pip

WORKDIR /graphdb/
COPY ./.bigfiles/graphdb-free-9.1.1-dist.zip .
RUN unzip -qq graphdb-free-9.1.1-dist.zip
ENV graphdb_home=/graphdb/graphdb-free-9.1.1
ENV PATH=${PATH}:${graphdb_home}/bin

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

WORKDIR /aida/ta2-pipeline/graphdb/
COPY ./env.sh .
RUN chmod u+x ./env.sh

WORKDIR /aida/ta2-pipeline/

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN pip install ipykernel && python -m ipykernel install --user && pip install -r requirements.txt 

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen
RUN locale-gen

#CMD [ "/bin/bash", "" ]
ENTRYPOINT ["/bin/bash", "/aida/ta2-pipeline/entrypoint.sh"]
