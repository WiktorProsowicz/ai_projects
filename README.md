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

- build C++ libs

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
python project_setup.py install_dependencies -o setup_mode=dev
```

- install pre-commit configuration in order to enable checks before committing

```bash
pre-commit install

# alternatively run pre-commit once on every file
pre-commit run --all-files [hook_id]
```
