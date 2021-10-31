FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04

RUN apt update

RUN pip install awscli
RUN pip install boto3

RUN pip install cython
RUN git clone https://github.com/ifzhang/FairMOT /fairmot
RUN cd /fairmot && pip install -r requirements.txt

RUN git clone https://github.com/jinfagang/DCNv2_latest.git /DCNv2

COPY resources/FairMOT/make.sh /DCNv2/make.sh
COPY resources/FairMOT/opts.py /fairmot/src/lib/opts.py
COPY resources/FairMOT/track.py /fairmot/src/track.py
COPY resources/FairMOT/pose_dla_dcn.py /fairmot/src/lib/models/networks/pose_dla_dcn.py
COPY resources/FairMOT/model.py /fairmot/src/lib/models/model.py
COPY resources/FairMOT/multitracker.py /fairmot/src/lib/tracker/multitracker.py
COPY resources/FairMOT/config.py /fairmot/src/lib/config.py

##########################################################################################
# SageMaker requirements
##########################################################################################
## install flask
RUN pip install flask
RUN pip install gunicorn

### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
 
COPY resources/*.* /
ENV WORKDIR /

ENTRYPOINT ["python", "/serve.py"]
