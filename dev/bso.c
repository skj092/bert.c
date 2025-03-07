#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float *load_tensor(const char *bin_path, int *total_size) {
  FILE *bin_file = fopen(bin_path, "rb");
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
  float *data = (float *)malloc(file_size);
  if (!data) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    fclose(bin_file);
    return NULL;
  }

  size_t read_count = fread(data, 1, file_size, bin_file);
  if (read_count != file_size) {
    fprintf(stderr, "Warning: Expected to read %ld bytes, but got %ld\n",
            file_size, read_count);
  }

  fclose(bin_file);
  return data;
}
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
  const char *dense_weight_path = "bins/tmp/qw.bin";
  const char *dense_bias_path = "bins/tmp/kw.bin";
  const char *ln_w_path_path = "bins/tmp/vw.bin";
  const char *ln_b_path_path = "bins/tmp/vw.bin";

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


// Example usage
int main() {
  // Initialize configuration
  BertConfig config = default_config();

  BertSelfOutput bert_output;

  const char *bert_output_path = "bins/att_out.bin";

  // Load model weights - if shape files don't exist, we'll infer the shape
  // float *out = NULL;
  // out = load_tensor(bert_output_path, &total_size);

  // Initialize model (load weight and biases)
  load_from_checkpoint(&bert_output, config);

  // Forward pass
  // bert_model_forward(model, input_ids, token_type_ids, batch_size,
  // seq_length, sequence_output, pooled_output);

  // free_bert_model(model);

  return 0;
}
