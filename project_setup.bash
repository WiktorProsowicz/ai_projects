#!/bin/bash

PROJECT_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" || return; pwd);

function build_project()
{
    cmake -S "${PROJECT_HOME}" -B "${PROJECT_HOME}/build" "$@";
    cd "${PROJECT_HOME}/build" || return;
    make -j $(( $(grep -c ^processor /proc/cpuinfo) / 2 ));
    cd ..;
    
}

function run_cppcheck()
{
    cppcheck --enable=all --project=build/compile_commands.json --check-config -iForeignModules/* --std=c++20 --suppress=missingIncludeSystem -ibuild/* .;
}

function rebuild_project()
{
    cd "${PROJECT_HOME}" || return;
    rm -rf "${PROJECT_HOME}/build";
    mkdir -p "${PROJECT_HOME}/build";
    build_project "$@";
}

function rebuild_python_environment()
{
    rm -rf "${PROJECT_HOME}/venv"
    python3 -m venv venv
    source "$PROJECT_HOME/venv/bin/activate"
    python3 -m pip install -r $PROJECT_HOME/requirements.txt
    deactivate
}

function run_clang_tidy()
{
    local paths=$(find "${PROJECT_HOME}" \
                \( -name "*.cpp" -o -name "*.c" -o -name "*.cc" -o -name "*.hpp" \) \
                -not \( -path "${PROJECT_HOME}/ForeignModules/*" -prune \) \
                -not \( -path "${PROJECT_HOME}/build/*" -prune \) \
                -print );

    for path in $paths; do

        [[ $# != 0 ]] && [[ ! "${path}" =~ $1 ]] && continue

        printf "\033[1;34m\nChecking path '$path':\n\033[0m"

        clang-tidy-14 -p "${PROJECT_HOME}/build" -config-file="${PROJECT_HOME}/.clang-tidy" -extra-arg=-std=c++20 -header-filter=".*" $path &> .cltid__

        local output=$(cat .cltid__ | wc -l)

        if [[ $output == 3 ]]
        then
            printf "\033[1;32mOK\033[0m\n"
        else
            cat .cltid__
        fi

        rm .cltid__
    done
}

function apply_clang_format()
{
    local paths=$(find "${PROJECT_HOME}" \
                \( -name "*.cpp" -o -name "*.c" -o -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) \
                -not \( -path "${PROJECT_HOME}/ForeignModules/*" -prune \) \
                -not \( -path "${PROJECT_HOME}/build/*" -prune \) \
                -print );

    for path in $paths; do
        
        clang-format -n $path &> .clfor__

        local output=$(cat .clfor__ | wc -l)

        if [[ $output != 0 ]]
        then
            printf "Changed file \033[1;33m$path\033[0m\n"
            clang-format -i $path
        fi

        rm .clfor__
    done
}

function run_tests()
{
    ctest --test-dir "${PROJECT_HOME}/build" --output-on-failure
}

function download_dependencies
{
    cd "${PROJECT_HOME}"/ForeignModules || return
    
    git clone --recursive "git@github.com:google/googletest.git"
    git clone --recursive "https://github.com/fmtlib/fmt.git"
    
    cd ..
}