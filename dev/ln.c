#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>


// Helper function to print an array of floats
void print_float_array(float *array, size_t size) {
  for (size_t i = 0; i < size; i++) {
    printf("%.6f ", array[i]);
  }
  printf("\n");
}


// Kahan summation for more accurate floating-point summation
double kahan_sum(const float *arr, int n) {
    double sum = 0.0;
    double c = 0.0;  // A running compensation for lost low-order bits
    for (int i = 0; i < n; i++) {
        double y = (double)arr[i] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

void layernorm_forward(float *out, float *inp, float *weight, float *bias,
                       int B, int T, int C) {
    // Use more precise epsilon
    const double eps = 1e-12;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // Pointer to current input slice
            float *x = inp + b * T * C + t * C;

            // Compute mean using Kahan summation
            double mean = kahan_sum(x, C) / C;

            // Compute variance
            double variance = 0.0;
            double c = 0.0;
            for (int i = 0; i < C; i++) {
                double xshift = (double)x[i] - mean;
                double y = xshift * xshift - c;
                double t = variance + y;
                c = (t - variance) - y;
                variance = t;
            }
            variance /= C;

            // Compute reciprocal standard deviation
            double rstd = 1.0 / sqrt(variance + eps);

            // Pointer to current output slice
            float *out_bt = out + b * T * C + t * C;

            // Normalize, scale, and shift
            for (int i = 0; i < C; i++) {
                // Compute normalized value using high precision
                double normalized = ((double)x[i] - mean) * rstd;

                // Scale and shift
                out_bt[i] = (float)(normalized * weight[i] + bias[i]);
            }
        }
    }
}

// Enhanced binary file loader with more robust error checking
float* load_binary_file(const char *filename, size_t *num_elements) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
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
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed for %s\n", filename);
        fclose(file);
        *num_elements = 0;
        return NULL;
    }

    // Read file contents
    size_t elements_read = fread(data, sizeof(float), *num_elements, file);
    fclose(file);

    if (elements_read != *num_elements) {
        fprintf(stderr, "Error: Could not read entire file %s\n", filename);
        free(data);
        *num_elements = 0;
        return NULL;
    }

    return data;
}

// Enhanced tensor comparison with more detailed reporting
int compare_tensors(const float *ref, const float *test, size_t n, double tolerance) {
    int total_elements = n;
    int matching_elements = 0;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    int first_mismatch_index = -1;

    for (size_t i = 0; i < n; i++) {
        double abs_diff = fabs((double)ref[i] - (double)test[i]);
        double rel_diff = (abs_diff == 0.0) ? 0.0 :
                           abs_diff / fmax(fabs((double)ref[i]), fabs((double)test[i]));

        max_abs_diff = fmax(max_abs_diff, abs_diff);
        max_rel_diff = fmax(max_rel_diff, rel_diff);

        if (abs_diff <= tolerance) {
            matching_elements++;
        } else if (first_mismatch_index == -1) {
            first_mismatch_index = i;
        }
    }

    printf("Tensor Comparison Results:\n");
    printf("Total Elements: %d\n", total_elements);
    printf("Matching Elements: %d (%.2f%%)\n",
           matching_elements,
           (matching_elements * 100.0) / total_elements);
    printf("Max Absolute Difference: %e\n", max_abs_diff);
    printf("Max Relative Difference: %e\n", max_rel_diff);

    if (first_mismatch_index != -1) {
        printf("First Mismatch at Index %d\n", first_mismatch_index);
        printf("Reference: %f, Test: %f\n",
               ref[first_mismatch_index],
               test[first_mismatch_index]);
    }

    return (matching_elements == total_elements);
}

int main() {
    // Dimensions
    const int B = 2;    // batch
    const int T = 128;  // time / sequence length
    const int C = 768;  // number of channels

    // Tolerance for floating-point comparisons
    const double TOLERANCE = 1e-4;

    // Allocate memory for tensors
    size_t tensor_size = B * T * C;
    size_t channel_size = C;

    float *x = NULL;     // input
    float *w = NULL;     // weights
    float *b = NULL;     // biases
    float *ref_out = NULL;  // reference output
    size_t elements;

    // Load input tensors
    x = load_binary_file("bins/ln_i.bin", &elements);
    w = load_binary_file("bins/ln_w.bin", &elements);
    b = load_binary_file("bins/ln_b.bin", &elements);
    ref_out = load_binary_file("bins/ln_o.bin", &elements);

    if (!x || !w || !b || !ref_out) {
        fprintf(stderr, "Failed to load all required binary files\n");
        goto cleanup;
    }

    // Allocate output tensor
    float *c_out = (float *)malloc(tensor_size * sizeof(float));
    if (!c_out) {
        fprintf(stderr, "Memory allocation failed for output tensor\n");
        goto cleanup;
    }

    // Perform LayerNorm forward pass
    layernorm_forward(c_out, x, w, b, B, T, C);

    // Compare results (checking first 100 elements as an example)
    int result = compare_tensors(ref_out, c_out, 100, TOLERANCE);

    printf("\nLayer Normalization Comparison: %s\n",
           result ? "PASSED" : "FAILED");

cleanup:
    // Free allocated memory
    free(x);
    free(w);
    free(b);
    free(ref_out);
    free(c_out);

    return result ? 0 : 1;
}
