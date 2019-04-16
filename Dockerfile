FROM ubuntu:16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion gcc g++

RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb

RUN apt-get install -y autoconf automake autoconf-archive pkg-config libxml2-dev \
    libicu-dev libboost-dev libboost-regex-dev \
    libtar-dev libbz2-dev zlib1g-dev libexttextcat-dev cython3 && apt-get clean    

RUN pip install -U spacy
RUN pip install -U gensim
RUN pip install pyldavis
RUN pip install librosa
RUN python -m spacy download en
RUN python -m spacy download nl
RUN python -m spacy download de

RUN mkdir -p /usr/src/app
RUN mkdir -p /usr/src/libs

WORKDIR /usr/src/libs

ADD https://github.com/LanguageMachines/ticcutils/releases/download/v0.18/ticcutils-0.18.tar.gz /usr/src/libs/
RUN tar -xvzf ticcutils-0.18.tar.gz && rm -rf ./ticcutils-0.18.tar.gz

ADD https://github.com/LanguageMachines/libfolia/releases/download/v1.12/libfolia-1.12.tar.gz /usr/src/libs/
RUN tar -xvzf libfolia-1.12.tar.gz && rm -rf ./libfolia-1.12.tar.gz

ADD https://github.com/LanguageMachines/uctodata/releases/download/v0.6/uctodata-0.6.tar.gz /usr/src/libs/
RUN tar -xvzf uctodata-0.6.tar.gz && rm -rf ./uctodata-0.6.tar.gz

ADD https://github.com/LanguageMachines/ucto/releases/download/v0.12/ucto-0.12.tar.gz /usr/src/libs/
RUN tar -xvzf ucto-0.12.tar.gz && rm -rf ./ucto-0.12.tar.gz

ADD https://github.com/LanguageMachines/timbl/releases/download/v6.4.11/timbl-6.4.11.tar.gz /usr/src/libs/
RUN tar -xvzf timbl-6.4.11.tar.gz && rm -rf ./timbl-6.4.11.tar.gz

ADD https://github.com/LanguageMachines/mbt/releases/download/v3.3.1/mbt-3.3.1.tar.gz /usr/src/libs/
RUN tar -xvzf mbt-3.3.1.tar.gz && rm -rf ./mbt-3.3.1.tar.gz

ADD https://github.com/LanguageMachines/frogdata/releases/download/v0.15/frogdata-0.15.tar.gz /usr/src/libs/
RUN tar -xvzf frogdata-0.15.tar.gz && rm -rf ./frogdata-0.15.tar.gz

ADD https://github.com/LanguageMachines/frog/releases/download/v0.14/frog-0.14.tar.gz /usr/src/libs/
RUN tar -xvzf frog-0.14.tar.gz && rm -rf ./frog-0.14.tar.gz

ADD https://github.com/proycon/python-frog/archive/v0.3.7.tar.gz /usr/src/libs/
RUN tar -xvzf v0.3.7.tar.gz && rm -rf ./v0.3.7.tar.gz

WORKDIR /usr/src/libs/ticcutils-0.18
RUN ./configure && make && make install && ldconfig && make check

WORKDIR /usr/src/libs/libfolia-1.12
RUN bash bootstrap.sh && ./configure && make && make install && ldconfig && make check

WORKDIR /usr/src/libs/uctodata-0.6
RUN bash bootstrap.sh && ./configure && make && make install && ldconfig

WORKDIR /usr/src/libs/ucto-0.12
RUN bash bootstrap.sh && ./configure && make && make install && ldconfig && make check

WORKDIR /usr/src/libs/timbl-6.4.11
RUN bash bootstrap.sh && ./configure && make && make install && ldconfig && make check

WORKDIR /usr/src/libs/mbt-3.3.1
RUN bash bootstrap.sh && ./configure && make && make install && ldconfig && make check

WORKDIR /usr/src/libs/frogdata-0.15
RUN bash bootstrap.sh && ./configure && make && make install && ldconfig && make check

WORKDIR /usr/src/libs/frog-0.14
RUN bash bootstrap.sh && ./configure && make && make install && ldconfig && make check

WORKDIR /usr/src/libs/python-frog-0.3.7
RUN python setup.py install

WORKDIR /usr/src/libs/
RUN git clone https://github.com/cltl/OpenDutchWordnet.git

RUN echo 'export PYTHONPATH="${PYTHONPATH}:/usr/src/libs/"' >> /root/.bashrc 

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

WORKDIR /usr/src/app