set positional-arguments

# ---------------------------------------------------------------------------
# Standard recipes.
# ---------------------------------------------------------------------------

# Set up an environment for installing dependencies and running scripts.
setup_venv:
    uv python install 3.12
    uv venv

# Install main project-wide dependencies (e.g. build tools, cpp libraries, etc.).
install_build_tools:
    @echo "Installing project dependencies..."
    uv pip install -r pyproject.toml

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

# ---------------------------------------------------------------------------
# Development recipes.
# ---------------------------------------------------------------------------

# Install development dependencies (e.g. linters, formatters).
dev_install_dev_tools:
    @echo "Installing development dependencies..."
    uv pip install -r pyproject.toml --extra dev

# Copy .vscode settings template to the main project path.
dev_setup_vscode_settings:
    @echo "Setting up VSCode settings..."
    cp setuputils/.vscode/settings.json .vscode/settings.json

# ---------------------------------------------------------------------------
# CI recipes.
# ---------------------------------------------------------------------------

# Install tools for running CI scripts.
ci_install_ci_tools:
    @echo "Installing CI tools..."
    uv pip install -r pyproject.toml --extra ci

# Run tests using ctest and collects coverage stats.
ci_run_tests:
    @echo "Running tests..."
    rm -rf test_results/ && mkdir -p test_results/coverage
    uv run ctest -T Test --test-dir build --output-on-failure
    uv run gcovr -r . --gcov-executable gcov-14 \
        -o test_results/coverage/coverage.html \
        --fail-under-line 80 --html-details \
        --filter "cpplibs/.*" \
        --exclude ".*/tests/.*"

# Run static checks for repository, such as pre-commit hooks and clang-tidy.
ci_run_precommit:
    #!/usr/bin/env bash

    echo "Running pre-commit checks..."

    uv run pre-commit run --all-files

# Run clang tidy static checks.
ci_run_clang_tidy:
    just --justfile {{justfile()}} build_project -DENABLE_CLANG_TIDY=ON

# Run include-what-you-use static checks.
ci_run_iwyu_checks:
    #!/usr/bin/env bash
    just --justfile {{justfile()}} build_project -DENABLE_IWYU=ON | tee /tmp/ci_iwyu_output
    found_warnings=$(grep -E "Warning: include-what-you-use reported diagnostics:" /tmp/ci_iwyu_output)

    if [[ $found_warnings ]]; then
        exit 1
    fi


# Run cppcheck static checks.
ci_run_cppcheck:
    just --justfile {{justfile()}} build_project -DENABLE_CPPCHECK=ON
