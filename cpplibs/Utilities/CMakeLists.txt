project(Utilities VERSION 1.0.0)

aiprojects_add_library(LIBRARY_NAME Utilities)

add_tests(LINKED_LIBRARIES LoggingLib fmt::fmt)
