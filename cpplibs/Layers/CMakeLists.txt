project(Layers VERSION 1.0.0)

aiprojects_add_library(LIBRARY_NAME Layers)

target_link_libraries(Layers Interfaces MLCore LoggingLib)

add_tests(LINKED_LIBRARIES LoggingLib Utilities fmt::fmt)
