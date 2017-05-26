# build this with `docker build -t geomdata/builder -f .dockerfile .`
# then push it with `docker push geomdata/builder`
FROM continuumio/anaconda:latest
ADD requirements.txt requirements.txt
RUN apt-get -q -y update
RUN conda create -y --quiet --name gda_env --file requirements.txt python=3
RUN apt-get -q -y upgrade
RUN apt-get -q -y install gcc
RUN apt-get -q -y install --no-install-recommends texlive-latex-recommended texlive-latex-extra texlive-bibtex-extra texlive-formats-extra texlive-math-extra texlive-pictures texlive-pstricks texlive-publishers texlive-xetex texlive-luatex texlive-fonts-recommended texlive-generic-extra texlive-generic-recommended
RUN apt-get -q -y clean
RUN tl-paper set all letter
RUN conda create -y --quiet --name gda_env2 --file requirements.txt python=2
