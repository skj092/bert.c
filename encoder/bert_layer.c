#include "../attention/utils.c"
#include "bert_utils.c"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Self-attention component
typedef struct {
  float *query_weight; // [hidden_size, hidden_size]
  float *query_bias;   // [hidden_size]
  float *key_weight;   // [hidden_size, hidden_size]
  float *key_bias;     // [hidden_size]
  float *value_weight; // [hidden_size, hidden_size]
  float *value_bias;   // [hidden_size]
  int num_heads;
  int head_dim;
  float scale;
} BertSelfAttention;

void load_from_checkpoint(BertSelfAttention *bsa, BertConfig config) {
  const char *query_weight_path = "bins/tmp/qw.bin";
  const char *key_weight_path = "bins/tmp/kw.bin";
  const char *value_weight_path = "bins/tmp/vw.bin";

  // Paths to bias files
  const char *query_bias_path = "bins/tmp/qb.bin";
  const char *key_bias_path = "bins/tmp/kb.bin";
  const char *value_bias_path = "bins/tmp/vb.bin";
  float *query_weight = NULL;
  float *key_weight = NULL;
  float *value_weight = NULL;
  float *query_bias = NULL;
  float *key_bias = NULL;
  float *value_bias = NULL;

  // Try to load with shape files first
  int total_size;
  query_weight = load_tensor(query_weight_path, &total_size);
  key_weight = load_tensor(key_weight_path, &total_size);
  value_weight = load_tensor(value_weight_path, &total_size);
  query_bias = load_tensor(query_bias_path, &total_size);
  key_bias = load_tensor(key_bias_path, &total_size);
  value_bias = load_tensor(value_bias_path, &total_size);
  printf("model weights and biases loaded..\n");

  bsa->query_weight = query_weight;
  bsa->query_bias = query_bias;
  bsa->key_weight = key_weight;
  bsa->key_bias = key_bias;
  bsa->value_weight = value_weight;
  bsa->value_bias = value_bias;
}

void bert_attention_forward(BertSelfAttention BertSelfAttention,
                            float *input_tensor, float *attn_output) {
  // Allocate memory for intermediate results
  int batch_size = 2;
  int seq_length = 128;
  int hidden_size = 768;
  int num_heads = 12;
  int head_dim = hidden_size / num_heads;

  const int total_seq_hidden = batch_size * seq_length * hidden_size;
  float *q_proj = (float *)malloc(total_seq_hidden * sizeof(float));
  float *k_proj = (float *)malloc(total_seq_hidden * sizeof(float));
  float *v_proj = (float *)malloc(total_seq_hidden * sizeof(float));
  // Linear projections
  printf("Computing linear projections...\n");
  linear_projection(input_tensor, BertSelfAttention.query_weight,
                    BertSelfAttention.query_bias, q_proj, batch_size,
                    seq_length, hidden_size, hidden_size);
  linear_projection(input_tensor, BertSelfAttention.key_weight,
                    BertSelfAttention.key_bias, k_proj, batch_size, seq_length,
                    hidden_size, hidden_size);
  linear_projection(input_tensor, BertSelfAttention.value_weight,
                    BertSelfAttention.value_bias, v_proj, batch_size,
                    seq_length, hidden_size, hidden_size);
  // Reshape for multi-head attention
  printf("Reshaping for multi-head attention...\n");
  const int total_head_seq_dim = batch_size * num_heads * seq_length * head_dim;
  float *q_reshaped = (float *)malloc(total_head_seq_dim * sizeof(float));
  float *k_reshaped = (float *)malloc(total_head_seq_dim * sizeof(float));
  float *v_reshaped = (float *)malloc(total_head_seq_dim * sizeof(float));

  reshape_for_multihead(q_proj, q_reshaped, batch_size, seq_length, num_heads,
                        head_dim);
  reshape_for_multihead(k_proj, k_reshaped, batch_size, seq_length, num_heads,
                        head_dim);
  reshape_for_multihead(v_proj, v_reshaped, batch_size, seq_length, num_heads,
                        head_dim);

  // Compute attention scores (Q*K^T) for each batch and head
  printf("Computing attention scores...\n");
  const int attn_size = batch_size * num_heads * seq_length * seq_length;
  float *attn_scores = (float *)malloc(attn_size * sizeof(float));
  float *attn_scaled = (float *)malloc(attn_size * sizeof(float));
  float *attn_softmax = (float *)malloc(attn_size * sizeof(float));

  // Scale factor for attention
  float scale_factor = 1.0f / sqrtf(head_dim);

  // For each batch and head, compute attention
  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < num_heads; h++) {
      float *q_ptr = q_reshaped + (b * num_heads + h) * seq_length * head_dim;
      float *k_ptr = k_reshaped + (b * num_heads + h) * seq_length * head_dim;
      float *v_ptr = v_reshaped + (b * num_heads + h) * seq_length * head_dim;
      float *scores_ptr =
          attn_scores + (b * num_heads + h) * seq_length * seq_length;
      float *scaled_ptr =
          attn_scaled + (b * num_heads + h) * seq_length * seq_length;
      float *softmax_ptr =
          attn_softmax + (b * num_heads + h) * seq_length * seq_length;

      // Compute Q*K^T
      matmul_transpose(q_ptr, k_ptr, scores_ptr, seq_length, head_dim,
                       seq_length);

      // Apply scaling
      for (int i = 0; i < seq_length * seq_length; i++) {
        scaled_ptr[i] = scores_ptr[i] * scale_factor;
      }

      // Apply softmax
      softmax(scaled_ptr, softmax_ptr, seq_length, seq_length);
    }
  }
  printf("Computing attention output...\n");
  // float *attn_output = (float *)malloc(total_head_seq_dim * sizeof(float));

  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < num_heads; h++) {
      float *softmax_ptr =
          attn_softmax + (b * num_heads + h) * seq_length * seq_length;
      float *v_ptr = v_reshaped + (b * num_heads + h) * seq_length * head_dim;
      float *output_ptr =
          attn_output + (b * num_heads + h) * seq_length * head_dim;

      // Compute softmax * V
      matmul(softmax_ptr, v_ptr, output_ptr, seq_length, seq_length, head_dim);
    }
  }
  // Reshape back to original dimensions
  printf("Reshaping output and saving final results...\n");
  float *context = (float *)malloc(total_seq_hidden * sizeof(float));

  reshape_from_multihead(attn_output, context, batch_size, seq_length,
                         num_heads, head_dim);
}

int main() {
  // BERT configuration
  BertConfig config = default_config();
  const int batch_size = 2;
  const int seq_length = 128;
  const int hidden_size = 768;
  const int num_heads = 12;
  const int head_dim = hidden_size / num_heads;
  const int layer_idx = 0;
  const int total_seq_hidden = batch_size * seq_length * hidden_size;

  // Load Self Attention Layer
  BertSelfAttention SelfAttention;
  load_from_checkpoint(&SelfAttention, config);

  // self attn output
  const char *self_attn_output = "bins/att_out.bin";

  // Load model weights - if shape files don't exist, we'll infer the shape
  int weight_shape[2] = {hidden_size, hidden_size}; // Default shape for weights
  int bias_shape[1] = {hidden_size};                // Default shape for biases

  // Load input tensor
  printf("Loading input tensor from bins/temp.bin...\n");
  int input_size;
  float *input_tensor = load_tensor("bins/layer0_attn_input.bin", &input_size);
  print_float_array(input_tensor, 10);

  float *context = (float *)malloc(total_seq_hidden * sizeof(float));
  bert_attention_forward(SelfAttention, input_tensor, context);

  float *attn_out = NULL;
  int attn_size0;
  attn_out = load_tensor(self_attn_output, &attn_size0);
  check_tensor(context, attn_out, 10, "attnoutput");

  // Free the input_tensor loaded from file
  free(input_tensor);

  // Free the context buffer allocated in main
  free(context);

  // Free the attention output loaded for verification
  free(attn_out);

  // Free the SelfAttention model parameters
  free(SelfAttention.query_weight);
  free(SelfAttention.query_bias);
  free(SelfAttention.key_weight);
  free(SelfAttention.key_bias);
  free(SelfAttention.value_weight);
  free(SelfAttention.value_bias);

  return 0;
}
