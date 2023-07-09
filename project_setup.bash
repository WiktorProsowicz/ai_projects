#!/bin/bash

PROJECT_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" || return; pwd);

function build_project()
{
    cmake -S "${PROJECT_HOME}" -B "${PROJECT_HOME}/build" "$@";
    cd "${PROJECT_HOME}/build" || return;
    make;
    cd ..;
    cppcheck --enable=all --project=build/compile_commands.json --check-config -iForeignModules/* --std=c++11 --suppress=missingIncludeSystem -ibuild/* .;

}

function rebuild_project()
{
    cd "${PROJECT_HOME}" || return;
    rm -rf "${PROJECT_HOME}/build";
    mkdir -p "${PROJECT_HOME}/build";
    build_project "$@";
}

function run_clang_tidy()
{
    find "${PROJECT_HOME}" \
        \( -name "*.cpp" -o -name "*.c" -o -name "*.cc" \) \
        -not \( -path "${PROJECT_HOME}/ForeignModules/*" -prune \) \
        -not \( -path "${PROJECT_HOME}/build/*" -prune \) \
        -exec clang-tidy -p "${PROJECT_HOME}/build" -config-file="${PROJECT_HOME}/.clang-tidy" -extra-arg=-std=c++20 -header-filter=".*" {} \;

}

function run_tests()
{
    ctest --test-dir "${PROJECT_HOME}/build" --output-on-failure 
}
