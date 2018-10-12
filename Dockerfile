# Get appropriate CUDA and cuDNN images
FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# Change home directory
WORKDIR /home

# Install dependencies
ADD install-deps /home/install-deps
RUN ./install-deps

# Change working directory
WORKDIR /home/style-transfer

# Install python requirements
ADD requirements.txt /home/style-transfer/requirement.txt
RUN pip3 install -r requirement.txt

# Add files
ADD . /home/style-transfer
