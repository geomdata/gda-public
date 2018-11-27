FROM registry.gitlab.com/geomdata/gda-open/builder
USER fakeuser
RUN git clone https://github.com/geomdata/gda-public.git
RUN git clone https://github.com/scipy/scipy-sphinx-theme
RUN cd gda-public/doc_src &&\
    ln -sf /home/fakeuser/scipy-sphinx-theme/_theme ./ &&\
 	cd .. &&\
 	ls -lahrt doc_src/_theme/
WORKDIR /home/fakeuser/gda-public
RUN source activate gda_env3 &&\
    python setup.py build_ext --inplace &&\
    py.test &&\
    python setup.py build_doc_html -EW &&\
    source deactivate
