# Changelog

LoggingLib is a module containing classes used to handle various kinds of logged messages. Logging is crucial for proper tracking and debugging of the program execution. Additionally collected logs can be used as the history of run tests, trained models etc.

## 1.0.0

- Introduced `Logger`
- Introduced `Stream Wrappers`(#stream-wrappers)
    - `IStreamWrapper`
    - `BaseStreamWrapper`
    - `DecolorizingStream`
- Introduced new macros
    - `LOG_RESET_LOGGER`
    - `LOG_SET_DEFAULT_STREAM`
    - `LOG_SET_NAMED_STREAM`

## 0.0.1

- Introduced macros
    - `LOG_WARN`
    - `LOG_ERROR`
    - `LOG_INFO`
