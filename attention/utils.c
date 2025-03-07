#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper function to print an array of floats
void print_float_array(float *array, size_t size) {
  for (size_t i = 0; i < size; i++) {
    printf("%.6f ", array[i]);
  }
  printf("\n");
}

// Function declarations
void print_int_array(int *weights, size_t num_values) {
  for (size_t i = 0; i < num_values; i++) {
    printf("%d ", weights[i]);
    if ((i + 1) % 8 == 0)
      printf("\n");
  }
  printf("\n");
}


// poor man's tensor checker
int check_tensor(float *a, float *b, int n, const char *label) {
  int print_upto = 4;
  int ok = 1;
  float maxdiff = 0.0f;
  float tol = 2e-2f;
  printf("%s\n", label);
  for (int i = 0; i < n; i++) {
    // look at the diffence at position i of these two tensors
    float diff = fabsf(a[i] - b[i]);

    // keep track of the overall error
    ok = ok && (diff <= tol);
    if (diff > maxdiff) {
      maxdiff = diff;
    }

    // for the first few elements of each tensor, pretty print
    // the actual numbers, so we can do a visual, qualitative proof/assessment
    if (i < print_upto) {
      if (diff <= tol) {
        if (i < print_upto) {
          printf("OK ");
        }
      } else {
        if (i < print_upto) {
          printf("NOT OK ");
        }
      }
      printf("%f %f\n", a[i], b[i]);
    }
  }
  // print the final result for this tensor
  if (ok) {
    printf("TENSOR OK, maxdiff = %e\n", maxdiff);
  } else {
    printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
  }
  return ok;
}


// Load tensor without shape file (we'll provide the shape directly)
float* load_tensor(const char* bin_path, int* total_size) {
    FILE* bin_file = fopen(bin_path, "rb");
    if (!bin_file) {
        fprintf(stderr, "Error: Could not open file %s\n", bin_path);
        return NULL;
    }

    // Get file size
    fseek(bin_file, 0, SEEK_END);
    long file_size = ftell(bin_file);
    fseek(bin_file, 0, SEEK_SET);

    *total_size = file_size / sizeof(float);

    // Allocate memory and read data
    float* data = (float*)malloc(file_size);
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(bin_file);
        return NULL;
    }

    size_t read_count = fread(data, 1, file_size, bin_file);
    if (read_count != file_size) {
        fprintf(stderr, "Warning: Expected to read %ld bytes, but got %ld\n", file_size, read_count);
    }

    fclose(bin_file);
    return data;
}

void matmul_forward_naive(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}




void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    const int LOOP_UNROLL = 8;
    if (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel for
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for (int o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            float result[LOOP_UNROLL];
            // initialize the bias, if it exists
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            for (int i = 0; i < C; i++) {
                float w = weight[i + o * C];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                int bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

// Apply element-wise addition
void add_tensors(float *out, const float *a, const float *b, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}

// Apply dropout (for inference mode, this is just a copy operation)
void apply_dropout(float *out, const float *inp, float dropout_prob, int size) {
    // During inference, dropout is identity function
    if (dropout_prob == 0.0f) {
        memcpy(out, inp, size * sizeof(float));
        return;
    }

    // During training (not implemented for simplicity)
    memcpy(out, inp, size * sizeof(float));
}

void layernorm_forward_cpu(float *out, const float *inp, const float *weight,
                           const float *bias, int B, int T, int C, float eps) {
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the input position inp[b,t,:]
      const float *x = inp + b * T * C + t * C;
      // calculate the mean
      float m = 0.0f;
      for (int i = 0; i < C; i++) {
        m += x[i];
      }
      m = m / C;
      // calculate the variance (without any bias correction)
      float v = 0.0f;
      for (int i = 0; i < C; i++) {
        float xshift = x[i] - m;
        v += xshift * xshift;
      }
      v = v / C;
      // calculate the rstd
      float s = 1.0f / sqrtf(v + eps);
      // seek to the output position in out[b,t,:]
      float *out_bt = out + b * T * C + t * C;
      for (int i = 0; i < C; i++) {
        float n = (s * (x[i] - m));        // normalized output
        float o = n * weight[i] + bias[i]; // scale and shift it
        out_bt[i] = o;                     // write
      }
    }
  }
}

