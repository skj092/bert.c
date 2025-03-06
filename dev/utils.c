#include <stdio.h>

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

