FROM continuumio/anaconda3
RUN conda create -n py3.7 python=3.7
RUN echo 'conda activate py3.7' >> ~/.bashrc
COPY ./requirements.txt /tmp/requirements.txt
RUN conda install -n py3.7 cudatoolkit=9.0 
RUN conda install -n py3.7 pip
RUN /opt/conda/envs/py3.7/bin/python -m pip install -r /tmp/requirements.txt
