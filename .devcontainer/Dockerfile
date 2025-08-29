# ------------------------------------------------------------------------
# This file contains the definition of a docker image used in course
# of the development process. The image built from this config contains
# the basic dependencies one needs in order to be able to setup the whole
# environment without installing anything locally on the system.
# ------------------------------------------------------------------------

FROM mcr.microsoft.com/devcontainers/base:noble
ARG DEBIAN_FRONTEND=noninteractive

# Installing the main packages required for a new project's user
# They're necessary to correctly use the project's content
RUN apt update && \
    # Main dependencies
    apt update && apt install -y \
        just gcc-14 g++-14 wget

USER vscode
RUN wget -qO- https://astral.sh/uv/install.sh | sh 
