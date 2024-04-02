"""Script containing functions for managing the project workspace.

Available functions and corresponding arguments are described in the
main function as well as in the doc strings of the functions.
"""
import argparse
import json
import logging
import os
import pathlib
import re
import subprocess
import venv
from os import path

from pymodules.utilities import logging_utils

HOME_PATH = pathlib.Path(__file__).absolute().parent.as_posix()


def _run_clang_tidy():
    """Runs clang-tidy analysis on C++ files present in the repository.

    Throws:
        subprocess.CalledProcessError: If the clang-tidy tool crashed.
    """

    build_path = os.path.join(HOME_PATH, 'build')
    compile_db_path = os.path.join(build_path, 'compile_commands.json')

    if not os.path.exists(compile_db_path):
        raise FileNotFoundError('Could not find compile database.')

    files_to_check = []

    with open(compile_db_path, encoding='utf-8') as compilation_db_fp:

        compile_db = json.load(compilation_db_fp)

        for entry in compile_db:
            files_to_check.append(entry['file'])

    tidy_config_path = os.path.join(HOME_PATH, 'setuputils', '.clang-tidy')

    tidy_args = (
        '-p',
        build_path,
        f'--config-file={tidy_config_path}',
        '--use-color',
        '-extra-arg=-Wno-unknown-warning-option'
    )

    subprocess.run(['clang-tidy', *tidy_args,
                    *files_to_check], check=True)


def setup_venv():
    """Sets up the virtual environment."""

    venv_path = path.join(HOME_PATH, 'venv')

    if os.path.exists(venv_path):
        logging.warning(  # pylint: disable=logging-not-lazy
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
        '-DCMAKE_CXX_COMPILER=g++-12'
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


def install_dependencies(*args):
    """Downloads and builds external libraries.

    The used libraries are specified by the `conanfile.py`. The function used Conan package manager
    executable. The user may provide custom options to the executable using syntax.
        -o OPTION_NAME=OPTION_VALUE
    The supported options are listed in the `conanfile.py` in the `options` class member.

    Args:
        *args: Variable number of command-line arguments. Acceptable are:
            - --build_debug: Flag telling whether the installed libs should be compiled as debug.
            - Arguments passed to the Conan executable. Profile-specifying args shall be omitted.

    Example:
        install_dependencies('--build_debug', '-v', '-o', 'setup_mode=dev')
    """

    conan_profile_re = re.compile('(-pr.*|--profile.*)')

    build_debug = any(map(lambda arg: arg == '--build_debug', args))

    args = filter(lambda arg: arg != '--build_debug', args)
    args = filter(lambda arg: not conan_profile_re.match(arg), args)

    default_args = ('--build=missing',)

    if build_debug:
        default_args += (
            '--profile:host=setuputils/conan/profile_debug.ini',
            '--profile:build=setuputils/conan/profile_debug.ini',
        )
    else:
        default_args += (
            '--profile:host=setuputils/conan/profile_release.ini',
            '--profile:build=setuputils/conan/profile_release.ini',
        )

    try:
        subprocess.run(
            ['conan', 'install', *default_args, *args, '.'], check=True)

    except subprocess.CalledProcessError as proc_error:
        logging.critical(
            'Setting up external dependencies failed: %s', proc_error)


def run_repository_checks():
    """Runs static checks on the files found in the repository.

    The purpose is to determine whether the current code is qualifies to be
    merged into main branch. This together with unit tests is an obligatory
    condition so that the code can be published on the main project branch.

    The checks include:
        - running pre-commit hooks specified in .pre-commit-config.yaml
        - running clang-tidy static analysis on C++ files
    """

    try:

        logging.info('Running pre-commit hooks.')

        subprocess.run(['pre-commit', 'run', '--all-files'], check=True)

        logging.info('Running clang-tidy checks.')

        _run_clang_tidy()

    except Exception:  # pylint: disable=broad-exception-caught
        logging.critical('Static checks failed!')


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

    def get_basic_doc(function) -> str:
        if function.__doc__:
            return function.__doc__.splitlines()[0]

        return ''

    functions_descriptions = '\n'.join(
        [f'{func.__name__}: {get_basic_doc(func)}' for func in [
            setup_venv, build_project, clean_project,
            install_dependencies, run_unit_tests, run_repository_checks]]
    )

    program_desc = (
        'Script contains functions helping with project management.\n'
        + 'Available functions:\n\n'
        + f'{functions_descriptions}'
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
                           install_dependencies, run_unit_tests, run_repository_checks]:
        if available_func.__name__ == function:
            available_func(*args)  # type: ignore
            return

    logging.error("Couldn't find the function '%s'.", function)


if __name__ == '__main__':

    parser = _get_arg_parser()
    arguments, left_args = parser.parse_known_args()

    logging_utils.setup_logging()

    main(arguments.function_name, *(arguments.args + left_args))
