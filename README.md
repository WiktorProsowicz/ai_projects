# ai_projects

This project is aimed at creating software components of a framework that helps developing AI models. The project's functionalities are related to the following concepts:

- providing API for building neural networks for deep learning
- sound data processing
- handling of complex computations involved in forward/backward model pass

## How to use

### Normal user

To use the project's functionalities one should follow this guide:

- ensure you have `python3.12` and `python3.12-venv` installed
- create virtual environment

```bash
python3.12 project_setup.py setup_venv
source venv/bin/activate
```

- install primary dependencies for either python scripts or project management

```bash
pip install -r requirements.txt
```

- install external C++ dependencies

```bash
python project_setup.py install_dependencies -o setup_mode=release
```

- build C++ libs (the default configuration uses `Unix Makefiles` generator, therefore ensure you have `make` installed)

```bash
# optionally clean the project in order to change the build configuration
python project_setup.py clean_project

python project_setup.py build_project [cmake_args]
```

### Developer

In order to properly contribute to the project, read the above guide and follow these steps:

- with activated venv install development-related dependencies

```bash
pip install -r requirements-dev.txt
python project_setup.py install_dependencies [--build_debug] -o setup_mode=dev
```

- install pre-commit configuration in order to enable checks before committing

```bash
pre-commit install

# alternatively run pre-commit once on every file
pre-commit run --all-files [hook_id]
```


### Run in container

An alternative way to use the project functionalities is by setting up a Docker container. This provides a stable project configuration and allows to run repository check pipelines, develop inside a container and run internal project scripts. This functionality assumes you have the Docker installed.

- build a Docker image from the defined Dockerfile

```bash
docker build -t ai_projects-release .
```

- instantiate a machine from the built image

```bash
docker run -dit --name=ai_projects-release-container ai_projects-release
```

- perform basic setup inside the repository

```bash
docker exec -it ai_projects-release-container /bin/bash

# inside the container
git clone https://github.com/WiktorProsowicz/ai_projects.git ai_projects && cd ai_projects && [git checkout SHA]
python3.12 project_setup.py setup_venv && source venv/bin/activate
pip install -r requirements.txt
python project_setup.py install_dependencies -o setup_mode=release
python project_setup.py build_project -DCMAKE_BUILD_TYPE=Release
```