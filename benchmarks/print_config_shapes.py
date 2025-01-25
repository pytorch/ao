from torchao.kernel import autotuner

configs = autotuner._load_best_configs()

print("m,k,n")
for k, v in configs.items():
    a_shape = k[1]
    b_shape = k[4]
    M, K0 = a_shape
    K1, N = b_shape

    assert K0 == K1

    print(f"{M},{K0},{N}")
