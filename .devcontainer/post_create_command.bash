#!/bin/bash

# This script should be run after the devcontainer is created.
# It's role is to establish the necessary environment for the
# user to start developing.

deactivate 2> /dev/null

rm -rf build && \
rm -rf .mypy_cache && \
rm -rf venv && \
rm -rf CMakeUserPresets.json && \
python3.12 project_setup.py setup_venv && source venv/bin/activate && \
pip install -r requirements.txt && pip install -r requirements-dev.txt && \
pip install clang-format==17.0.6 && \
python project_setup.py install_dependencies -o setup_mode=dev && \
python project_setup.py build_project -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
