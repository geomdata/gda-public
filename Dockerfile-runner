# build this with `docker build -t registry.gitlab.com/geomdata/gda-open/runner -f Dockerfile-runner .`
# then push it with `docker push registry.gitlab.com/geomdata/gda-open/runner`
FROM registry.gitlab.com/geomdata/gda-open/builder
#RUN export PATH="$HOME/miniconda/bin:$PATH" &&\
#    conda create --clone gda_env3 -y --quiet --name gda_run
RUN git clone https://github.com/geomdata/gda-public.git /home/fakeuser/gda-public &&\
    export PATH="$HOME/miniconda/bin:$PATH" &&\
	source activate gda_env3 &&\
	pip install file:///home/fakeuser/gda-public
EXPOSE 8888
CMD export PATH="$HOME/miniconda/bin:$PATH" && source activate gda_env3 && jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --notebook-dir gda-public/examples
