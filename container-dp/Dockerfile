FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update

RUN pip install awscli
RUN pip install boto3

RUN pip install cython
RUN git clone https://github.com/ifzhang/FairMOT /fairmot
RUN cd /fairmot && pip install -r requirements.txt

RUN git clone https://github.com/jinfagang/DCNv2_latest.git /DCNv2
RUN cd /DCNv2 && git reset --hard fa9b2fd740ced2a22e0e7e913c3bf3934fd08098

##########################################################################################
# SageMaker requirements
##########################################################################################
COPY resources/train.py /opt/ml/code/train.py
COPY resources/FairMOT/make.sh /DCNv2/make.sh
COPY resources/FairMOT/opts.py /fairmot/src/lib/opts.py
COPY resources/FairMOT/train.py /fairmot/src/train.py
COPY resources/FairMOT/track.py /fairmot/src/track.py
COPY resources/FairMOT/pose_dla_dcn.py /fairmot/src/lib/models/networks/pose_dla_dcn.py
COPY resources/FairMOT/model.py /fairmot/src/lib/models/model.py

ENV SAGEMAKER_PROGRAM train.py
