#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.c"

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


// Helper function to load tensor from bin file with separate shape file
float* load_tensor(const char* bin_path, const char* shape_path, int* shape, int max_dims) {
    FILE* bin_file = fopen(bin_path, "rb");
    if (!bin_file) {
        fprintf(stderr, "Error: Could not open file %s\n", bin_path);
        return NULL;
    }

    FILE* shape_file = fopen(shape_path, "r");
    if (!shape_file) {
        fprintf(stderr, "Error: Could not open shape file %s\n", shape_path);
        fclose(bin_file);
        return NULL;
    }

    // Read shape information
    char shape_str[256];
    fgets(shape_str, sizeof(shape_str), shape_file);
    fclose(shape_file);

    // Parse shape string
    char* token = strtok(shape_str, ",");
    int dim_count = 0;

    while (token != NULL && dim_count < max_dims) {
        shape[dim_count++] = atoi(token);
        token = strtok(NULL, ",");
    }

    // Calculate total size
    int total_size = 1;
    for (int i = 0; i < dim_count; i++) {
        total_size *= shape[i];
    }

    // Allocate memory and read data
    float* data = (float*)malloc(total_size * sizeof(float));
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(bin_file);
        return NULL;
    }

    size_t read_count = fread(data, sizeof(float), total_size, bin_file);
    if (read_count != total_size) {
        fprintf(stderr, "Warning: Expected to read %d elements, but got %ld\n", total_size, read_count);
        // Continue anyway, as some tensor files might be padded or we might not know the exact size
    }

    fclose(bin_file);
    return data;
}

// Load tensor without shape file (we'll provide the shape directly)
float* load_tensor_without_shape(const char* bin_path, int* total_size) {
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

// Matrix multiplication: C = A * B
void matmul(float* A, float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
}

// Matrix multiplication with transpose: C = A * B^T
void matmul_transpose(float* A, float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[j * n + l];  // Note B indices are swapped
            }
        }
    }
}

// Linear projection: Y = X * W^T + b
void linear_projection(float* X, float* W, float* b, float* Y, int batch_size, int seq_len, int input_dim, int output_dim) {
    for (int bs = 0; bs < batch_size; bs++) {
        for (int seq = 0; seq < seq_len; seq++) {
            float* x_ptr = X + (bs * seq_len + seq) * input_dim;
            float* y_ptr = Y + (bs * seq_len + seq) * output_dim;

            for (int i = 0; i < output_dim; i++) {
                y_ptr[i] = b[i];  // Add bias
                for (int j = 0; j < input_dim; j++) {
                    y_ptr[i] += x_ptr[j] * W[i * input_dim + j];
                }
            }
        }
    }
}

// Reshape tensor from [batch_size, seq_len, hidden_size] to [batch_size, num_heads, seq_len, head_dim]
void reshape_for_multihead(float* input, float* output, int batch_size, int seq_len, int num_heads, int head_dim) {
    int hidden_size = num_heads * head_dim;

    for (int bs = 0; bs < batch_size; bs++) {
        for (int h = 0; h < num_heads; h++) {
            for (int seq = 0; seq < seq_len; seq++) {
                for (int d = 0; d < head_dim; d++) {
                    int input_idx = (bs * seq_len + seq) * hidden_size + h * head_dim + d;
                    int output_idx = ((bs * num_heads + h) * seq_len + seq) * head_dim + d;
                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
}

// Reshape tensor from [batch_size, num_heads, seq_len, head_dim] back to [batch_size, seq_len, hidden_size]
void reshape_from_multihead(float* input, float* output, int batch_size, int seq_len, int num_heads, int head_dim) {
    int hidden_size = num_heads * head_dim;

    for (int bs = 0; bs < batch_size; bs++) {
        for (int seq = 0; seq < seq_len; seq++) {
            for (int h = 0; h < num_heads; h++) {
                for (int d = 0; d < head_dim; d++) {
                    int input_idx = ((bs * num_heads + h) * seq_len + seq) * head_dim + d;
                    int output_idx = (bs * seq_len + seq) * hidden_size + h * head_dim + d;
                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
}

// Softmax function for attention weights
void softmax(float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        // Find max for numerical stability
        float max_val = input[i * cols];
        for (int j = 1; j < cols; j++) {
            if (input[i * cols + j] > max_val) {
                max_val = input[i * cols + j];
            }
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = expf(input[i * cols + j] - max_val);
            sum += output[i * cols + j];
        }

        // Normalize
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] /= sum;
        }
    }
}

// Save tensor to a binary file
void save_tensor_to_bin(const char* filename, float* tensor, int size) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return;
    }

    fwrite(tensor, sizeof(float), size, file);
    fclose(file);
}

int main() {
    // BERT configuration
    const int batch_size = 2;
    const int seq_length = 128;
    const int hidden_size = 768;
    const int num_heads = 12;
    const int head_dim = hidden_size / num_heads;
    const int layer_idx = 0;  // Assuming we're processing the first layer

    // self attn output
    const char* self_attn_output = "bins/att_out.bin";
    // Paths to weight files
    const char* query_weight_path = "bins/tmp/qw.bin";
    const char* key_weight_path = "bins/tmp/kw.bin";
    const char* value_weight_path = "bins/tmp/vw.bin";

    // Paths to bias files
    const char* query_bias_path = "bins/tmp/qb.bin";
    const char* key_bias_path = "bins/tmp/kb.bin";
    const char* value_bias_path = "bins/tmp/vb.bin";

    // Paths to shape files
    const char* query_weight_shape_path = "bins/weights/encoder_layer_0_attention_self_query_weight.bin.shape";
    const char* key_weight_shape_path = "bins/weights/encoder_layer_0_attention_self_key_weight.bin.shape";
    const char* value_weight_shape_path = "bins/weights/encoder_layer_0_attention_self_value_weight.bin.shape";
    const char* query_bias_shape_path = "bins/weights/encoder_layer_0_attention_self_query_bias.bin.shape";
    const char* key_bias_shape_path = "bins/weights/encoder_layer_0_attention_self_key_bias.bin.shape";
    const char* value_bias_shape_path = "bins/weights/encoder_layer_0_attention_self_value_bias.bin.shape";

    // Load model weights - if shape files don't exist, we'll infer the shape
    int weight_shape[2] = {hidden_size, hidden_size};  // Default shape for weights
    int bias_shape[1] = {hidden_size};                // Default shape for biases

    float* query_weight = NULL;
    float* key_weight = NULL;
    float* value_weight = NULL;
    float* query_bias = NULL;
    float* key_bias = NULL;
    float* value_bias = NULL;

    // Try to load with shape files first
    printf("Loading model weights and biases...\n");

    FILE* test_shape_file = fopen(query_weight_shape_path, "r");
    if (test_shape_file) {
        // Shape files exist, use them
        fclose(test_shape_file);
        query_weight = load_tensor(query_weight_path, query_weight_shape_path, weight_shape, 2);
        key_weight = load_tensor(key_weight_path, key_weight_shape_path, weight_shape, 2);
        value_weight = load_tensor(value_weight_path, value_weight_shape_path, weight_shape, 2);
        query_bias = load_tensor(query_bias_path, query_bias_shape_path, bias_shape, 1);
        key_bias = load_tensor(key_bias_path, key_bias_shape_path, bias_shape, 1);
        value_bias = load_tensor(value_bias_path, value_bias_shape_path, bias_shape, 1);
    } else {
        // Shape files don't exist, load without them (use default shapes)
        printf("Shape files not found. Using default tensor shapes.\n");

        int total_size;
        query_weight = load_tensor_without_shape(query_weight_path, &total_size);
        if (total_size != hidden_size * hidden_size) {
            printf("Warning: Expected weight size %d, got %d\n", hidden_size * hidden_size, total_size);
        }

        key_weight = load_tensor_without_shape(key_weight_path, &total_size);
        value_weight = load_tensor_without_shape(value_weight_path, &total_size);

        query_bias = load_tensor_without_shape(query_bias_path, &total_size);
        if (total_size != hidden_size) {
            printf("Warning: Expected bias size %d, got %d\n", hidden_size, total_size);
        }

        key_bias = load_tensor_without_shape(key_bias_path, &total_size);
        value_bias = load_tensor_without_shape(value_bias_path, &total_size);
    }

    // Check if all weights loaded
    if (!query_weight || !key_weight || !value_weight ||
        !query_bias || !key_bias || !value_bias) {
        fprintf(stderr, "Error: Failed to load weights and biases\n");
        // Free any allocated resources
        if (query_weight) free(query_weight);
        if (key_weight) free(key_weight);
        if (value_weight) free(value_weight);
        if (query_bias) free(query_bias);
        if (key_bias) free(key_bias);
        if (value_bias) free(value_bias);
        return 1;
    }

    // Load input tensor
    printf("Loading input tensor from bins/temp.bin...\n");
    int input_size;
    float* input_tensor = load_tensor_without_shape("bins/layer0_attn_input.bin", &input_size);
    print_float_array(input_tensor, 10);

    if (!input_tensor) {
        fprintf(stderr, "Error: Failed to load input tensor\n");
        // Free resources
        free(query_weight); free(query_bias); free(key_weight);
        free(key_bias); free(value_weight); free(value_bias);
        return 1;
    }

    // Verify input tensor size
    if (input_size != batch_size * seq_length * hidden_size) {
        printf("Warning: Expected input size %d, got %d. Adjusting dimensions...\n",
               batch_size * seq_length * hidden_size, input_size);

        // // Try to infer correct dimensions
        // if (input_size % hidden_size == 0) {
        //     int total_tokens = input_size / hidden_size;
        //     if (total_tokens % batch_size == 0) {
        //         seq_length = total_tokens / batch_size;
        //         printf("Adjusted seq_length to %d\n", seq_length);
        //     } else if (total_tokens % seq_length == 0) {
        //         batch_size = total_tokens / seq_length;
        //         printf("Adjusted batch_size to %d\n", batch_size);
        //     }
        // }
    }

    printf("Processing with dimensions: batch_size=%d, seq_length=%d, hidden_size=%d, num_heads=%d\n",
           batch_size, seq_length, hidden_size, num_heads);

    // Allocate memory for intermediate results
    const int total_seq_hidden = batch_size * seq_length * hidden_size;
    float* q_proj = (float*)malloc(total_seq_hidden * sizeof(float));
    float* k_proj = (float*)malloc(total_seq_hidden * sizeof(float));
    float* v_proj = (float*)malloc(total_seq_hidden * sizeof(float));

    if (!q_proj || !k_proj || !v_proj) {
        fprintf(stderr, "Error: Memory allocation failed for projections\n");
        free(query_weight); free(query_bias); free(key_weight);
        free(key_bias); free(value_weight); free(value_bias);
        free(input_tensor);
        if (q_proj) free(q_proj);
        if (k_proj) free(k_proj);
        if (v_proj) free(v_proj);
        return 1;
    }

    // Linear projections
    printf("Computing linear projections...\n");
    linear_projection(input_tensor, query_weight, query_bias, q_proj, batch_size, seq_length, hidden_size, hidden_size);
    linear_projection(input_tensor, key_weight, key_bias, k_proj, batch_size, seq_length, hidden_size, hidden_size);
    linear_projection(input_tensor, value_weight, value_bias, v_proj, batch_size, seq_length, hidden_size, hidden_size);

    // Save the linear projections
    save_tensor_to_bin("bins/c_layer0_q_proj.bin", q_proj, total_seq_hidden);
    save_tensor_to_bin("bins/c_layer0_k_proj.bin", k_proj, total_seq_hidden);
    save_tensor_to_bin("bins/c_layer0_v_proj.bin", v_proj, total_seq_hidden);

    // Reshape for multi-head attention
    printf("Reshaping for multi-head attention...\n");
    const int total_head_seq_dim = batch_size * num_heads * seq_length * head_dim;
    float* q_reshaped = (float*)malloc(total_head_seq_dim * sizeof(float));
    float* k_reshaped = (float*)malloc(total_head_seq_dim * sizeof(float));
    float* v_reshaped = (float*)malloc(total_head_seq_dim * sizeof(float));

    if (!q_reshaped || !k_reshaped || !v_reshaped) {
        fprintf(stderr, "Error: Memory allocation failed for reshaped tensors\n");
        free(query_weight); free(query_bias); free(key_weight);
        free(key_bias); free(value_weight); free(value_bias);
        free(input_tensor); free(q_proj); free(k_proj); free(v_proj);
        if (q_reshaped) free(q_reshaped);
        if (k_reshaped) free(k_reshaped);
        if (v_reshaped) free(v_reshaped);
        return 1;
    }

    reshape_for_multihead(q_proj, q_reshaped, batch_size, seq_length, num_heads, head_dim);
    reshape_for_multihead(k_proj, k_reshaped, batch_size, seq_length, num_heads, head_dim);
    reshape_for_multihead(v_proj, v_reshaped, batch_size, seq_length, num_heads, head_dim);

    // Save reshaped tensors
    save_tensor_to_bin("bins/c_layer0_q_reshaped.bin", q_reshaped, total_head_seq_dim);
    save_tensor_to_bin("bins/c_layer0_k_reshaped.bin", k_reshaped, total_head_seq_dim);
    save_tensor_to_bin("bins/c_layer0_v_reshaped.bin", v_reshaped, total_head_seq_dim);

    // Compute attention scores (Q*K^T) for each batch and head
    printf("Computing attention scores...\n");
    const int attn_size = batch_size * num_heads * seq_length * seq_length;
    float* attn_scores = (float*)malloc(attn_size * sizeof(float));
    float* attn_scaled = (float*)malloc(attn_size * sizeof(float));
    float* attn_softmax = (float*)malloc(attn_size * sizeof(float));

    if (!attn_scores || !attn_scaled || !attn_softmax) {
        fprintf(stderr, "Error: Memory allocation failed for attention matrices\n");
        free(query_weight); free(query_bias); free(key_weight);
        free(key_bias); free(value_weight); free(value_bias);
        free(input_tensor); free(q_proj); free(k_proj); free(v_proj);
        free(q_reshaped); free(k_reshaped); free(v_reshaped);
        if (attn_scores) free(attn_scores);
        if (attn_scaled) free(attn_scaled);
        if (attn_softmax) free(attn_softmax);
        return 1;
    }

    // Scale factor for attention
    float scale_factor = 1.0f / sqrtf(head_dim);

    // For each batch and head, compute attention
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            float* q_ptr = q_reshaped + (b * num_heads + h) * seq_length * head_dim;
            float* k_ptr = k_reshaped + (b * num_heads + h) * seq_length * head_dim;
            float* v_ptr = v_reshaped + (b * num_heads + h) * seq_length * head_dim;
            float* scores_ptr = attn_scores + (b * num_heads + h) * seq_length * seq_length;
            float* scaled_ptr = attn_scaled + (b * num_heads + h) * seq_length * seq_length;
            float* softmax_ptr = attn_softmax + (b * num_heads + h) * seq_length * seq_length;

            // Compute Q*K^T
            matmul_transpose(q_ptr, k_ptr, scores_ptr, seq_length, head_dim, seq_length);

            // Apply scaling
            for (int i = 0; i < seq_length * seq_length; i++) {
                scaled_ptr[i] = scores_ptr[i] * scale_factor;
            }

            // Apply softmax
            softmax(scaled_ptr, softmax_ptr, seq_length, seq_length);
        }
    }

    // Save attention matrices
    save_tensor_to_bin("bins/c_layer0_attn_qk_product.bin", attn_scores, attn_size);
    save_tensor_to_bin("bins/c_layer0_attn_scaled.bin", attn_scaled, attn_size);
    save_tensor_to_bin("bins/c_layer0_attn_softmax.bin", attn_softmax, attn_size);

    // Compute attention output (softmax * V) for each batch and head
    printf("Computing attention output...\n");
    float* attn_output = (float*)malloc(total_head_seq_dim * sizeof(float));

    if (!attn_output) {
        fprintf(stderr, "Error: Memory allocation failed for attention output\n");
        free(query_weight); free(query_bias); free(key_weight);
        free(key_bias); free(value_weight); free(value_bias);
        free(input_tensor); free(q_proj); free(k_proj); free(v_proj);
        free(q_reshaped); free(k_reshaped); free(v_reshaped);
        free(attn_scores); free(attn_scaled); free(attn_softmax);
        return 1;
    }

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            float* softmax_ptr = attn_softmax + (b * num_heads + h) * seq_length * seq_length;
            float* v_ptr = v_reshaped + (b * num_heads + h) * seq_length * head_dim;
            float* output_ptr = attn_output + (b * num_heads + h) * seq_length * head_dim;

            // Compute softmax * V
            matmul(softmax_ptr, v_ptr, output_ptr, seq_length, seq_length, head_dim);
        }
    }

    // Save attention output
    save_tensor_to_bin("bins/c_layer0_attn_output.bin", attn_output, total_head_seq_dim);

    // Reshape back to original dimensions
    printf("Reshaping output and saving final results...\n");
    float* context = (float*)malloc(total_seq_hidden * sizeof(float));

    if (!context) {
        fprintf(stderr, "Error: Memory allocation failed for context\n");
        free(query_weight); free(query_bias); free(key_weight);
        free(key_bias); free(value_weight); free(value_bias);
        free(input_tensor); free(q_proj); free(k_proj); free(v_proj);
        free(q_reshaped); free(k_reshaped); free(v_reshaped);
        free(attn_scores); free(attn_scaled); free(attn_softmax);
        free(attn_output);
        return 1;
    }

    reshape_from_multihead(attn_output, context, batch_size, seq_length, num_heads, head_dim);
    print_float_array(attn_output, 10);

    // Save final output
    save_tensor_to_bin("bins/c_att_out.bin", context, total_seq_hidden);

    // Print output dimensions
    printf("Output shape: [%d, %d, %d]\n", batch_size, seq_length, hidden_size);

    // verify with torch output
    // const char* self_attn_output = "bins/att_out.bin";
    // load_tensor_without_shape(const char *bin_path, int *total_size)
    float* attn_out = NULL;
    int attn_size0;
    attn_out = load_tensor_without_shape(self_attn_output, &attn_size0);
    check_tensor(attn_out, attn_output, 10, "attnoutput");

    // Free all allocated memory
    printf("Freeing resources...\n");
    free(query_weight);
    free(query_bias);
    free(key_weight);
    free(key_bias);
    free(value_weight);
    free(value_bias);
    free(input_tensor);
    free(q_proj);
    free(k_proj);
    free(v_proj);
    free(q_reshaped);
    free(k_reshaped);
    free(v_reshaped);
    free(attn_scores);
    free(attn_scaled);
    free(attn_softmax);
    free(attn_output);
    free(context);

    printf("BERT self-attention computation completed successfully!\n");
    return 0;
}
