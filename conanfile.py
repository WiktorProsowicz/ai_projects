"""Contains class responsible for managing project C++ dependencies."""
import os

import conan  # type: ignore, pylint: disable=import-error


class AiProjectsRecipe(conan.ConanFile):
    """Specifies the project's requirements for external dependencies."""

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
