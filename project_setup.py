# -*- coding: utf-8 -*-
"""Script containing functions for managing the project workspace.

Available functions and corresponding arguments are described in the
main function as well as in the doc strings of the functions.
"""
import argparse
import logging
import os
import pathlib
import subprocess
import venv
from os import path

from pymodules.utilities import logging_utils

HOME_PATH = pathlib.Path(__file__).absolute().parent.as_posix()


def setup_venv():
    """Sets up the virtual environment."""

    venv_path = path.join(HOME_PATH, 'venv')

    if os.path.exists(venv_path):
        logging.warning(
            "Directory '%s' already exists. If you are sure you want to"
            + ' replace it with a new environment, delete it and run again.',
            venv_path,
        )
        return

    venv.create(venv_path, with_pip=True, upgrade_deps=True, clear=False)

    logging.info(
        "Successfully created a virtual environment at directory '%s'", venv_path
    )
    logging.info(
        "You can now activate the environment with 'source ./venv/bin/activate'."
    )
    logging.info(
        "Then type 'python3 -m pip install -r requirements.txt' to install dependencies."
    )
    logging.info("Then type 'deactivate' to deactivate the environment.")


def build_project(*args):
    """Compiles C++ files from the repository.

    The compiled binaries are linked to each other and in form of libraries and executables
    are moved to `build` directory.

    Args:
        *args: Arguments passed to cmake while compiling.
            For supported arguments see CMakeLists.txt -> options definitions.
    """

    build_path = os.path.join(HOME_PATH, 'build')

    default_args = (
        '-S', HOME_PATH,
        '-B', build_path,
        '-DCMAKE_C_COMPILER=gcc-12',
        '-DCMAKE_CXX_COMPILER=g++-11'
    )

    try:
        subprocess.run(['cmake', *default_args, *args], check=True)

        subprocess.run(['make', '-C', build_path, '-j'], check=True)

    except subprocess.CalledProcessError as proc_error:
        logging.critical('Building project failed: %s', proc_error)


def clean_project():
    """Clears generated CMake configuration files.

    The project after having been cleaned is prepared to be rebuilt.
    """

    build_path = os.path.join(HOME_PATH, 'build')

    try:
        subprocess.run(
            ['rm', '-rf', f'{build_path}/CMakeFiles', f'{build_path}/CMakeCache.txt'], check=True)

    except subprocess.CalledProcessError as proc_error:
        logging.critical('Cleaning project failed: %s', proc_error)


def install_dependencies():
    """Downloads and builds external libraries.

    The used libraries are specified by the `conanfile.py`.
    """

    build_path = os.path.join(HOME_PATH, 'build')

    default_args = (
        f'--output-folder={build_path}/ConanFiles',
        '--build=missing',
        '--profile=./setuputils/conan_release_prof.ini'
    )

    try:
        subprocess.run(['conan', 'install', *default_args, '.'], check=True)

    except subprocess.CalledProcessError as proc_error:
        logging.critical(
            'Setting up external dependencies failed: %s', proc_error)


def run_unit_tests():
    """Runs C++ and Python tests found in the workspace."""

    build_path = os.path.join(HOME_PATH, 'build')

    try:

        logging.info('Running C++ unit tests.')

        subprocess.run(['ctest', '--test-dir', build_path,
                       '--output-on-failure'], check=True)

    except subprocess.CalledProcessError as proc_error:
        logging.critical(
            'Unit test run failed: %s', proc_error
        )


def _get_arg_parser() -> argparse.ArgumentParser:
    """Returns an argument parser for the script."""

    functions_descriptions = '\n'.join(
        [f"{func.__name__}: {func.__doc__.splitlines()[0]}" for func in [
            setup_venv, build_project, clean_project, install_dependencies, run_unit_tests]]
    )

    program_desc = (
        'Script contains functions helping with project management.\n'
        + 'Available functions:\n\n'
        + f"{functions_descriptions}"
    )

    arg_parser = argparse.ArgumentParser(
        description=program_desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    arg_parser.add_argument(
        'function_name', help='name of the function to be used')
    arg_parser.add_argument(
        'args', nargs='*', help='positional arguments for the function'
    )

    return arg_parser


def main(function: str, *args):
    """Main function delegating the flow to other ones.

    Args:
        function (str): Name of the function to be called.
    """

    for available_func in [setup_venv, build_project, clean_project,
                           install_dependencies, run_unit_tests]:
        if available_func.__name__ == function:
            available_func(*args)  # type: ignore
            return

    logging.error("Couldn't find the function '%s'.", function)


if __name__ == '__main__':

    parser = _get_arg_parser()
    arguments, left_args = parser.parse_known_args()

    logging_utils.setup_logging()

    main(arguments.function_name, *(arguments.args + left_args))
