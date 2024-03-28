# -----------------------------------------------------------------------------
# This file contains the definition of a docker image used to run the compiled
# executables and scripts defined in the project public api. During the image
# building process a basic setup of project's requirements is performed as well
# as the installation of project's dependencies. 
# -----------------------------------------------------------------------------

FROM ubuntu:22.04
WORKDIR /app
COPY . .
ARG DEBIAN_FRONTEND=noninteractive

# Installing the main packages required for a new project's user
# They're necessary to correctly use the project's content
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && apt install -y \
        python3.12 python3.12-venv

# Installing the internal project's dependencies
RUN python3.12 project_setup.py setup_venv && \
    source venv/bin/activate && \
    pip install -r requirements.txt &&
    
