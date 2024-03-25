"""Contains class responsible for managing project C++ dependencies."""

import conan


class AiProjectsRecipe(conan.ConanFile):
    """Specifies the project's requirements for external dependencies."""

    name = 'ai_projects'
    version = "1.0.0"
    author = 'WiktorProsowicz'
    url = "https://github.com/WiktorProsowicz/ai_projects"
    settings = "os", "arch", "compiler", "build_type"
    generators = 'CMakeDeps', 'CMakeToolchain'

    def requirements(self):
        """Specifies the basic project requirements."""

        self.requires('fmt/10.2.1')

    def layout(self):
        """Specifies the project building behavior.

        The behavior depends on the context in which the project is being built.
        """
