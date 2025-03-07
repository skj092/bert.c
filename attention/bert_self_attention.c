#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.c"

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
// Self-attention component
typedef struct {
    float *query_weight;  // [hidden_size, hidden_size]
    float *query_bias;    // [hidden_size]
    float *key_weight;    // [hidden_size, hidden_size]
    float *key_bias;      // [hidden_size]
    float *value_weight;  // [hidden_size, hidden_size]
    float *value_bias;    // [hidden_size]
    int num_heads;
    int head_dim;
    float scale;
} BertSelfAttention;



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
    int total_size;
    query_weight = load_tensor(query_weight_path, &total_size);
    if (total_size != hidden_size * hidden_size) {
        printf("Warning: Expected weight size %d, got %d\n", hidden_size * hidden_size, total_size);
    }

    key_weight = load_tensor(key_weight_path, &total_size);
    value_weight = load_tensor(value_weight_path, &total_size);

    query_bias = load_tensor(query_bias_path, &total_size);
    if (total_size != hidden_size) {
        printf("Warning: Expected bias size %d, got %d\n", hidden_size, total_size);
    }

    key_bias = load_tensor(key_bias_path, &total_size);
    value_bias = load_tensor(value_bias_path, &total_size);

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
    float* input_tensor = load_tensor("bins/layer0_attn_input.bin", &input_size);
    print_float_array(input_tensor, 10);

    if (!input_tensor) {
        fprintf(stderr, "Error: Failed to load input tensor\n");
        // Free resources
        free(query_weight); free(query_bias); free(key_weight);
        free(key_bias); free(value_weight); free(value_bias);
        return 1;
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

    // Print output dimensions
    printf("Output shape: [%d, %d, %d]\n", batch_size, seq_length, hidden_size);

    // verify with torch output
    float* attn_out = NULL;
    int attn_size0;
    attn_out = load_tensor(self_attn_output, &attn_size0);
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

