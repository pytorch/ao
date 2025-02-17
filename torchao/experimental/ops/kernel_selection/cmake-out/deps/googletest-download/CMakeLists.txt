CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)

PROJECT(googletest-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(googletest
	URL https://github.com/google/googletest/archive/release-1.11.0.zip
	URL_HASH SHA256=353571c2440176ded91c2de6d6cd88ddd41401d14692ec1f99e35d013feda55a
	SOURCE_DIR "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/deps/googletest"
	BINARY_DIR "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/cmake-out/deps/googletest"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)
