set positional-arguments

# Set up an environment for installing dependencies and running scripts
setup_venv:
    uv python install 3.12
    uv venv
    
# Install main project-wide dependencies (e.g. build tools, cpp libraries, etc.)
install_build_tools:
    @echo "Installing project dependencies..."
    uv pip install -r pyproject.toml

# Install development dependencies (e.g. linters, formatters)
install_dev_tools:
    @echo "Installing development dependencies..."
    uv pip install -r pyproject.toml --extra dev

build_project MODE="Release" BUILD_TESTS="False" ENABLE_TRACY="False":
    #!/usr/bin/env bash

    echo "Building the project..."

    tests_option=""
    tracy_option=""

    if [ "$2" = "true" ]; then tests_option="-DBUILD_TESTS=ON"; fi

    if [ "$3" = "true" ]; then tracy_option="-DENABLE_TRACY=ON"; fi

    uv run cmake -S . -B build -G Ninja \
        -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
        -DCMAKE_BUILD_TYPE=$1 $tests_option $tracy_option

    cd build && uv run ninja



