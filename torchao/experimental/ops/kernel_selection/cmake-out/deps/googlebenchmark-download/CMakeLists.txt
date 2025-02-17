CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)

PROJECT(googlebenchmark-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(googlebenchmark
	URL https://github.com/google/benchmark/archive/v1.6.1.zip
	URL_HASH SHA256=367e963b8620080aff8c831e24751852cffd1f74ea40f25d9cc1b667a9dd5e45
	SOURCE_DIR "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/deps/googlebenchmark"
	BINARY_DIR "/Users/scroy/repos/ao/torchao/experimental/ops/kernel_selection/cmake-out/deps/googlebenchmark"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)
