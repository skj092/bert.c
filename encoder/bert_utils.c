#include <stdbool.h>
#include <stddef.h>
#include <string.h>


typedef struct {
  bool is_decoder;
  int vocab_size;
  int hidden_size;
  int num_hidden_layers;
  int num_attention_heads;
  int intermediate_size;
  char hidden_act[10];
  float hidden_dropout_prob;
  float attention_probs_dropout_prob;
  int max_position_embeddings;
  float layer_norm_eps;
  int pad_token_id;
  int type_vocab_size;
} BertConfig;

// Default configuration for BERT base
BertConfig default_config() {
  BertConfig config;
  config.is_decoder = false;
  config.vocab_size = 30522;
  config.hidden_size = 768;
  config.num_hidden_layers = 12;
  config.num_attention_heads = 12;
  config.intermediate_size = 3072;
  strcpy(config.hidden_act, "gelu");
  config.hidden_dropout_prob = 0.1;
  config.attention_probs_dropout_prob = 0.1;
  config.max_position_embeddings = 512;
  config.layer_norm_eps = 1e-12;
  config.pad_token_id = 0;
  config.type_vocab_size = 2;
  return config;
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

