# -*- coding: utf-8 -*-
"""Script containing functions for managing the project workspace.

Available functions and corresponding arguments are described in the
main function as well as in the doc strings of the functions.
"""
import argparse
import logging
import os
import pathlib
import venv
from os import path

HOME_PATH = pathlib.Path(__file__).absolute().parent.as_posix()


def setup_venv() -> None:
    """Sets up the virtual environment."""

    venv_path = path.join(HOME_PATH, "venv")

    if os.path.exists(venv_path):
        logging.warning(
            "Directory '%s' already exists. If you are sure you want to"
            + " replace it with a new environment, delete it and run again.",
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


def _get_arg_parser() -> argparse.ArgumentParser:
    """Returns an argument parser for the script."""

    functions_descriptions = "\n".join(
        [f"{func.__name__}: {func.__doc__.splitlines()[0]}" for func in [
            setup_venv]]
    )

    program_desc = (
        "Script contains functions helping with project management.\n"
        + "Available functions:\n\n"
        + f"{functions_descriptions}"
    )

    arg_parser = argparse.ArgumentParser(
        description=program_desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    arg_parser.add_argument(
        "function_name", help="name of the function to be used")
    arg_parser.add_argument(
        "args", nargs="*", help="positional arguments for the function"
    )

    return arg_parser


def main(function: str, *args) -> None:
    """Main function delegating the flow to other ones.

    Args:
        function (str): Name of the function to be called.
    """

    for available_func in [setup_venv]:
        if available_func.__name__ == function:
            available_func(*args)  # type: ignore
            return

    logging.error("Couldn't find the function '%s'.", function)


if __name__ == "__main__":

    parser = _get_arg_parser()
    arguments = parser.parse_args()

    main(arguments.function_name, *arguments.args)
