# build this with `docker build -t registry.gitlab.com/geomdata/gda-open/builder -f Dockerfile-builder .`
# then push it with `docker push registry.gitlab.com/geomdata/gda-open/builder`
FROM registry.gitlab.com/geomdata/gda-open/basic:latest
USER root
ADD requirements.txt requirements.txt
RUN chown fakeuser requirements.txt && chmod 644 requirements.txt
USER fakeuser
RUN export PATH="$HOME/miniconda/bin:$PATH" &&\
    conda create -y --quiet --name gda_env3 --file requirements.txt python=3 &&\
	conda create -y --quiet --name gda_env2 --file requirements.txt python=2 &&\
	conda clean --all -y
RUN echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> /home/fakeuser/.bashrc
