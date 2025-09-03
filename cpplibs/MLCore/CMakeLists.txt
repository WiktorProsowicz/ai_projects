project(MLCore VERSION 1.0.0)

aiprojects_add_library(LIBRARY_NAME MLCore)

target_link_libraries(MLCore LoggingLib Utilities fmt::fmt)

add_tests()
