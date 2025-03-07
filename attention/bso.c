#include "utils.c"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
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

// Self-output component for attention
typedef struct {
  float *dense_weight;      // [hidden_size, hidden_size]
  float *dense_bias;        // [hidden_size]
  float *layer_norm_weight; // [hidden_size]
  float *layer_norm_bias;   // [hidden_size]
  float dropout_prob;
  float layer_norm_eps;
} BertSelfOutput;

void load_from_checkpoint(BertSelfOutput *bso, BertConfig config) {

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

void bert_output_forward(BertSelfOutput bert_self_output, float *output,
                         float *hidden_states, float *input_tensor, int B,
                         int T, int C) {

  // Temporary buffer for intermediate results
  float *temp_buffer = (float *)malloc(B * T * C * sizeof(float));
  if (!temp_buffer) {
    fprintf(stderr, "Error: Memory allocation failed for temp buffer\n");
    return;
  }

  // 1. Linear projection: hidden_states = self.dense(hidden_states)
  matmul_forward(temp_buffer, hidden_states, bert_self_output.dense_weight,
                 bert_self_output.dense_bias, B, T, C, C);

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
  print_float_array(residual, 10);

  // 4. Apply layer normalization: output = self.LayerNorm(residual)
  layernorm_forward_cpu(output, residual, bert_self_output.layer_norm_weight,
                        bert_self_output.layer_norm_bias, B, T, C,
                        bert_self_output.layer_norm_eps);

  // Print some values for debugging
  printf("Output after layer norm:\n");
  print_float_array(output, 10);

  // Free temporary buffers
  free(temp_buffer);
  free(residual);
}
// Example usage
int main() {
  // Initialize configuration
  BertConfig config = default_config();

  // Initialize model (load weight and biases)
  BertSelfOutput bert_self_output;
  load_from_checkpoint(&bert_self_output, config);

  const char *h_path = "bins/t1.bin";
  const char *x_path = "bins/tw.bin";
  const char *so_path = "bins/layer0_self_output_layernorm.bin";
  float *h = NULL;
  float *x = NULL;
  float *out = NULL;
  int out_size;
  float* c_out = (float*)malloc(2 * 128 * 768 * sizeof(float));
  int total_size;
  h = load_tensor(h_path, &total_size);
  x = load_tensor(x_path, &total_size);

  out = load_tensor(so_path, &total_size);

  bert_output_forward(bert_self_output, c_out, h, x, 2, 128, 768);
  check_tensor(c_out, out, 10, "bso");


  // Free all allocated memory
  free(h);
  free(x);
  free(out);
  free(bert_self_output.dense_weight);
  free(bert_self_output.dense_bias);
  free(bert_self_output.layer_norm_weight);
  free(bert_self_output.layer_norm_bias);

  printf("Execution completed successfully.\n");
  return 0;
}
