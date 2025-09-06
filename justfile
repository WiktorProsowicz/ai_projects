set positional-arguments

# Set up an environment for installing dependencies and running scripts.
setup_venv:
    uv python install 3.12
    uv venv

# Install main project-wide dependencies (e.g. build tools, cpp libraries, etc.).
install_build_tools:
    @echo "Installing project dependencies..."
    uv pip install -r pyproject.toml

# Install development dependencies (e.g. linters, formatters).
install_dev_tools:
    @echo "Installing development dependencies..."
    uv pip install -r pyproject.toml --extra dev

# Install tools for running Python scripts.
install_py_tools:
    @echo "Installing Python tools..."
    uv pip install -r pyproject.toml --extra py_scripts

# Build the project with specified options.
build_project *args="":
    #!/usr/bin/env bash

    echo "Building the project..."

    uv run cmake -S . -B build -G Ninja \
        -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 $@

    cd build && uv run ninja

# Remove the build directory and enable fresh build.
clean_project:
    @echo "Cleaning the project..."
    rm -rf build/

# Copy .vscode settings template to the main project path.
setup_vscode_settings:
    @echo "Setting up VSCode settings..."
    cp setuputils/.vscode/settings.json .vscode/settings.json

# Run tests using ctest.
run_tests:
    @echo "Running tests..."
    uv run ctest --test-dir build --output-on-failure

# Run static checks for repository, such as pre-commit hooks and clang-tidy.
run_static_checks file_regex="":
    #!/usr/bin/env bash

    echo "Running pre-commit checks..."

    uv run pre-commit run --all-files || exit 1

