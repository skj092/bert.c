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

// Self-output component for attention
typedef struct {
  float *dense_weight;      // [hidden_size, hidden_size]
  float *dense_bias;        // [hidden_size]
  float *layer_norm_weight; // [hidden_size]
  float *layer_norm_bias;   // [hidden_size]
  float dropout_prob;
  float layer_norm_eps;
} BertSelfOutput;

// Attention component
typedef struct {
  BertSelfAttention self;
  BertSelfOutput output;
} BertAttention;

void load_from_checkpoint1(BertSelfOutput *bso, BertConfig config) {

  // Paths to weight files
  const char *dense_weight_path =
      "bins/weights/encoder_layer_0_attention_output_dense_weight.bin";
  const char *dense_bias_path =
      "bins/weights/encoder_layer_0_attention_output_dense_bias.bin";
  const char *ln_w_path_path =
      "bins/weights/encoder_layer_0_attention_output_LayerNorm_weight.bin";
  const char *ln_b_path_path =
      "bins/weights/encoder_layer_0_attention_output_LayerNorm_bias.bin";

  // Load model weights - if shape files don't exist, we'll infer the shape
  int weight_shape[2] = {config.hidden_size, config.hidden_size};
  int bias_shape[1] = {config.hidden_size};

  float *dw = NULL;
  float *db = NULL;
  float *lnw = NULL;
  float *lnb = NULL;
  float *out = NULL;

  // Try to load with shape files first
  printf("Loading model weights and biases...\n");
  int total_size;
  dw = load_tensor(dense_weight_path, &total_size);
  db = load_tensor(dense_bias_path, &total_size);
  lnw = load_tensor(ln_w_path_path, &total_size);
  lnb = load_tensor(ln_b_path_path, &total_size);

  bso->dense_weight = dw;
  bso->dense_bias = db;
  bso->layer_norm_weight = lnw;
  bso->layer_norm_bias = lnb;
  bso->layer_norm_eps = config.layer_norm_eps;
  bso->dropout_prob = config.hidden_dropout_prob;
}

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
  // printf("model weights and biases loaded..\n");

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
  // printf("Computing linear projections...\n");
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
  // printf("Reshaping for multi-head attention...\n");
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
  // printf("Computing attention scores...\n");
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
  // printf("Computing attention output...\n");
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
  // printf("Reshaping output and saving final results...\n");
  float *context = (float *)malloc(total_seq_hidden * sizeof(float));

  reshape_from_multihead(attn_output, context, batch_size, seq_length,
                         num_heads, head_dim);
}

void bert_selfoutput_forward(BertSelfOutput bert_self_output, float *output,
                             float *hidden_states, float *input_tensor, int B,
                             int T, int C) {

  // Temporary buffer for intermediate results
  float *temp_buffer = (float *)malloc(B * T * C * sizeof(float));
  if (!temp_buffer) {
    fprintf(stderr, "Error: Memory allocation failed for temp buffer\n");
    return;
  }

  // print_float_array(temp_buffer, 5);
  print_float_array(hidden_states, 5);
  // print_float_array(bert_self_output.dense_weight, 5);
  // print_float_array(bert_self_output.dense_bias, 5);
  printf("Dimensions: B=%d T=%d C=%d C=%d\n", B, T, C, C);

  matmul_forward(temp_buffer, hidden_states, bert_self_output.dense_weight,
                 bert_self_output.dense_bias, B, T, C, C);

  print_float_array(temp_buffer, 5);
  // 2. Apply dropout (during inference, this is identity)
  apply_dropout(temp_buffer, temp_buffer, bert_self_output.dropout_prob,
                B * T * C);

  // 3. Add residual connection: residual = hidden_states + input_tensor
  float *residual = (float *)malloc(B * T * C * sizeof(float));
  if (!residual) {
    fprintf(stderr, "Error: Memory allocation failed for residual\n");
    free(temp_buffer);
    return;
  }
  add_tensors(residual, temp_buffer, input_tensor, B * T * C);
  // print_float_array(residual, 10);

  // 4. Apply layer normalization: output = self.LayerNorm(residual)
  layernorm_forward_cpu(output, residual, bert_self_output.layer_norm_weight,
                        bert_self_output.layer_norm_bias, B, T, C,
                        bert_self_output.layer_norm_eps);

  // Print some values for debugging
  printf("Output after layer norm:\n");
  // print_float_array(output, 10);

  // Free temporary buffers
  free(temp_buffer);
  free(residual);
}

void attention_forward(BertAttention attention, BertConfig configuration,
                       float *input, float *output) {

  bert_attention_forward(attention.self, input, output);
  // Verify attention output
  int total_size;
  const char *h_path = "bins/t1.bin";
  float *h = NULL;
  h = load_tensor(h_path, &total_size);
  check_tensor(h, output, 5, "ao");
  // bert_output_forward(BertSelfOutput bert_self_output, float *output, float
  // *hidden_states, float *input_tensor, int B, int T, int C)
  // print_float_array(output, 10);
  float *c_out = (float *)malloc(2 * 128 * 768 * sizeof(float));
  // bert_selfoutput_forward(attention.output, c_out, h, input, 2, 128, 768);
  bert_selfoutput_forward(attention.output, c_out, output, input, 2, 128, 768);
  print_float_array(c_out, 10);

  const char *so_path = "bins/layer0_self_output_layernorm.bin";
  float *out = NULL;
  out = load_tensor(so_path, &total_size);
  check_tensor(c_out, out, 5, "ao");
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

  BertSelfOutput bert_self_output;
  load_from_checkpoint1(&bert_self_output, config);

  BertAttention attention;
  attention.self = SelfAttention;
  attention.output = bert_self_output;

  const char *so_path = "bins/layer0_self_output_layernorm.bin";
  float *out = NULL;
  int total_size;
  int input_size;
  float *input_tensor = load_tensor("bins/layer0_attn_input.bin", &input_size);
  out = load_tensor(so_path, &total_size);
  // print_float_array(input_tensor, 10); // bert_pt.py :375

  float *context = (float *)malloc(total_seq_hidden * sizeof(float));
  attention_forward(attention, config, input_tensor, context);

  printf("loadng h and x\n");
  // print_float_array(context, 5);
  // print_float_array(input_tensor, 5);
  // print_float_array(bert_self_output.dense_bias, 5);
  // print_float_array(bert_self_output.dense_weight, 5);

  // bert_selfoutput_forward(bert_self_output, c_out, context, input_tensor, 2,
  //                         128, 768);
  // check_tensor(c_out, out, 10, "bso");

  // check_tensor(context, out, 10, "bso");

  free(context);

  // Free the attention output loaded for verification
  // free(attn_out);

  // Free the SelfAttention model parameters
  free(SelfAttention.query_weight);
  free(SelfAttention.query_bias);
  free(SelfAttention.key_weight);
  free(SelfAttention.key_bias);
  free(SelfAttention.value_weight);
  free(SelfAttention.value_bias);

  // Free all allocated memory
  // free(h);
  // free(x);
  free(out);
  free(bert_self_output.dense_weight);
  free(bert_self_output.dense_bias);
  free(bert_self_output.layer_norm_weight);
  free(bert_self_output.layer_norm_bias);

  return 0;
}
