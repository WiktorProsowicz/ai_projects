#!/bin/bash

PROJECT_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" || return; pwd);

function build_project() {
    cmake -S "${PROJECT_HOME}" -B "${PROJECT_HOME}/build" "$@";
    cd "${PROJECT_HOME}/build" || return;
    make;
    cd ..;
    cppcheck --enable=all --project=build/compile_commands.json --check-config -iForeignModules/* --std=c++11 --suppress=missingIncludeSystem -ibuild/* .;

    # find . \( -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) -exec clang-tidy -checks="*,-clang-diagnostic-error"  {} \;
}

function rebuild_project() {
    cd "${PROJECT_HOME}" || return;
    rm -rf "${PROJECT_HOME}/build";
    mkdir -p "${PROJECT_HOME}/build";
    build_project "$@";
}