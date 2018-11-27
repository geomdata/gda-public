# build this with `docker build -t registry.gitlab.com/geomdata/gda-open/basic -f Dockerfile-basic .`
# then push it with `docker push registry.gitlab.com/geomdata/gda-open/basic`
FROM debian:stable
SHELL ["/bin/bash", "-c"]
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get -q -y update --fix-missing &&\
    apt-get -q -y install gcc curl bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion xz-utils rsync &&\
    apt-get -q -y clean
RUN adduser fakeuser --shell /bin/bash --disabled-password --gecos "Fake User,0,0,0"
USER fakeuser
WORKDIR /home/fakeuser
RUN curl -s -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh;
RUN chmod 755 ./Miniconda3-latest-Linux-x86_64.sh &&\
	./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda &&\
    export PATH="$HOME/miniconda/bin:$PATH" &&\
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> .bashrc  &&\
    hash -r &&\
    conda config --set always_yes yes --set changeps1 yes &&\
    conda update -q conda &&\
    conda info -a &&\
	rm -fv ./Miniconda3-latest-Linux-x86_64.sh
