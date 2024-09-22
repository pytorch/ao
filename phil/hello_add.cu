__global__
void nothing_kernel(int* vectors, int* output) {
    
}

int main() {
    int N = 32;
    int vector_length = 3;
    size_t vectors_size = sizeof(int) * N * vector_length;
    int vectors[vectors_size];

    int out[vector_length];

    nothing_kernel<<<1, 1>>>(vectors, out);
}