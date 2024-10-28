all: test_lowbit

test_lowbit: test_lowbit.mm
	clang++ -I${TORCHAO_ROOT} -O3 -std=c++17  -Wall -Wextra -o $@ $< -framework Metal -framework Foundation

run: test_lowbit
	./test_lowbit
