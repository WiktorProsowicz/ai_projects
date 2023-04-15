#!/bin/bash

PROJECT_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" || return; pwd);

function build_project() {
    cmake -S "${PROJECT_HOME}" -B "${PROJECT_HOME}/build" "$@";
    cd "${PROJECT_HOME}/build" || return;
    make;
    cd ..;
    cppcheck --enable=all --project=build/compile_commands.json --check-config -iForeignModules/* --std=c++11 --suppress=missingIncludeSystem -ibuild/* .;

}

function rebuild_project() {
    cd "${PROJECT_HOME}" || return;
    rm -rf "${PROJECT_HOME}/build";
    mkdir -p "${PROJECT_HOME}/build";
    build_project "$@";
}

function run_clang_tidy() {
    find . -not -path "./ForeignModules/*" \( -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) \
        -exec clang-tidy -p "${PROJECT_HOME}/build" -checks="-abseil*,-altera*,-android*,bugprone*,cert*,-fuchsia*,-google*,-linuxkernel*,-llvm*,-llvmlibc*,modernize*,readability*"  {} \;

}