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

