// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void layernorm_forward(float *out, float *inp, float *weight, float *bias,
                       int B, int T, int C) {
  float eps = 1e-5f;
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the input position inp[b,t,:]
      float *x = inp + b * T * C + t * C;
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

// poor man's tensor checker
int check_tensor(float *a, float *b, int n, char *label) {
  int ok = 1;
  printf("%s\n", label);
  for (int i = 0; i < n; i++) {
    if (fabs(a[i] - b[i]) <= 1e-4) {
      printf("OK ");
    } else {
      printf("NOT OK ");
      ok = 0;
    }
    printf("%f %f\n", a[i], b[i]);
  }
  return ok;
}

float *load_binary_file(const char *filename, int *num_elements) {
  // Open the file
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    printf("Error opening file: %s\n", filename);
    *num_elements = 0;
    return NULL;
  }

  // Determine file size
  fseek(file, 0, SEEK_END);
  long file_size = ftell(file);
  rewind(file);

  // Calculate number of elements
  *num_elements = file_size / sizeof(float);

  // Allocate memory
  float *data = (float *)malloc(file_size);
  if (data == NULL) {
    printf("Memory allocation failed for file: %s\n", filename);
    fclose(file);
    *num_elements = 0;
    return NULL;
  }

  // Read file contents
  size_t elements_read = fread(data, sizeof(float), *num_elements, file);

  // Close file
  fclose(file);

  // Verify read operation
  if (elements_read != *num_elements) {
    printf("Error reading file: %s. Expected %d elements, read %zu\n", filename,
           *num_elements, elements_read);
    free(data);
    *num_elements = 0;
    return NULL;
  }

  return data;
}
void print_first_n_elements(float *data, int n, const char *array_name) {
  if (data == NULL) {
    printf("Cannot print elements: data is NULL\n");
    return;
  }

  printf("First %d elements of %s:\n", n, array_name);
  for (int i = 0; i < n; i++) {
    printf("%s[%d] = %f\n", array_name, i, data[i]);
  }
}

int main() {

  int B = 2;   // batch
  int T = 128; // time / sequence length
  int C = 768; // number of channels
  int num_elements;

  float *x = (float *)malloc(B * T * C * sizeof(float));
  float *w = (float *)malloc(C * sizeof(float));
  float *b = (float *)malloc(C * sizeof(float));
  float *out = (float *)malloc(B * T * C * sizeof(float));

  // Load data from python
  w = load_binary_file("bins/ln_w.bin", &num_elements);
  b = load_binary_file("bins/ln_b.bin", &num_elements);
  x = load_binary_file("bins/ln_i.bin", &num_elements);
  out = load_binary_file("bins/ln_o.bin", &num_elements);

  // forward pass
  float *c_out = (float *)malloc(B * T * C * sizeof(float));
  layernorm_forward(c_out, x, w, b, B, T, C);
  // check correctness of forward pass
  check_tensor(out, c_out, 10, "out");
  free(x);
  free(w);
  free(b);
  free(out);
  return 0;
}
