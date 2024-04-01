"""Contains class responsible for managing project C++ dependencies.

This file is used from the Conan executable level. For more info, see
project_setup.py::install_dependencies. 
"""
import os

import conan  # type: ignore, pylint: disable=import-error


class AiProjectsRecipe(conan.ConanFile):
    """Specifies the project's requirements for external dependencies.

    Attributes:
        options: Parameters telling how a single dependencies installing should be
            performed. Available options are:
                - setup_mode: Indicates whether the dependencies are being installed
                by a normal user or by a developer (In this case additional libs are required).  
    """

    name = 'ai_projects'
    version = '1.0.0'
    author = 'WiktorProsowicz'
    url = 'https://github.com/WiktorProsowicz/ai_projects'
    settings = 'os', 'arch', 'compiler', 'build_type'
    generators = 'CMakeDeps', 'CMakeToolchain'

    options = {'setup_mode': ['dev', 'release']}

    def requirements(self):
        """Specifies the basic project requirements."""

        self.requires('fmt/10.2.1')

        if self.options.setup_mode == 'dev':
            self.requires('gtest/1.14.0')
            self.requires('tracy/cci.20220130')

    def layout(self):
        """Specifies the project building behavior.

        The behavior depends on the context in which the project is being built.
        """

        self.folders.build = os.path.join('build')
        self.folders.generators = os.path.join('build', 'ConanFiles')
