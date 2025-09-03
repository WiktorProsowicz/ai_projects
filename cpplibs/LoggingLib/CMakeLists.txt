project(LoggingLib VERSION 1.0.0)

aiprojects_add_library(LIBRARY_NAME LoggingLib)

target_link_libraries(LoggingLib fmt::fmt)

add_tests()
