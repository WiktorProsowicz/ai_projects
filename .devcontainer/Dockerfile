# ------------------------------------------------------------------------
# This file contains the definition of a docker image used in course
# of the development process. The image built from this config contains
# the basic dependencies one needs in order to be able to setup the whole
# environment without installing anything locally on the system.   
# ------------------------------------------------------------------------

FROM mcr.microsoft.com/devcontainers/base:jammy
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && apt install -y \
        python3.12 python3.12-venv

