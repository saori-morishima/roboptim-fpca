ADD_EXECUTABLE(main main.cc)
PKG_CONFIG_USE_DEPENDENCY(main roboptim-core)
PKG_CONFIG_USE_DEPENDENCY(main roboptim-trajectory)