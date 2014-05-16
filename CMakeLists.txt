# Requires at least CMake 2.6 to configure the package.
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)

SET(PROJECT_NAME roboptim-fpca)
SET(PROJECT_DESCRIPTION "RobOptim trajectory")
SET(PROJECT_URL "http://github.com/saori-morishima/roboptim-fpca")

# FIXME: disable for now.
SET(CXX_DISABLE_WERROR 1)

SET(HEADERS
)

SETUP_PROJECT()

ADD_REQUIRED_DEPENDENCY("roboptim-core >= 2.0")
ADD_REQUIRED_DEPENDENCY("roboptim-trajectory")

HEADER_INSTALL("${HEADERS}")

ADD_SUBDIRECTORY(src)
#ADD_SUBDIRECTORY(tests)

SETUP_PROJECT_FINALIZE()
