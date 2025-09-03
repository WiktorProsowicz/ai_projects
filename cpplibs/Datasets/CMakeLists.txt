project(Datasets VERSION 1.0.0)

aiprojects_add_library(LIBRARY_NAME Datasets)

target_link_libraries(Datasets MLCore Interfaces)

add_tests(LINKED_LIBRARIES LoggingLib fmt::fmt)
