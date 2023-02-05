#!/bin/bash

PROJECT_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" || return; pwd);

function build_project() {
    cmake -S "${PROJECT_HOME}" -B "${PROJECT_HOME}/build" "$@";
    cd "${PROJECT_HOME}/build" || return;
    make install;
    cd ..;
}

function rebuild_project() {
    cd "${PROJECT_HOME}" || return;
    rm -rf "${PROJECT_HOME}/build";
    mkdir -p "${PROJECT_HOME}/build";
    build_project "$@";
}