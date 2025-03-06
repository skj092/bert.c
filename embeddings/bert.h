#ifndef BERT_H
#define BERT_H

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Define the BertConfig structure
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

// Define the BertEmbeddings structure
typedef struct {
  BertConfig *config;
  float *word_embeddings;
  float *position_embeddings;
  float *token_type_embeddings;
  float *layer_norm_weight;
  float *layer_norm_bias;
} BertEmbeddings;

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
// Function declarations
void print_int_array(int *weights, size_t num_values) {
  for (size_t i = 0; i < num_values; i++) {
    printf("%d ", weights[i]);
    if ((i + 1) % 8 == 0)
      printf("\n");
  }
  printf("\n");
}

// Define the is_training function
bool is_training() { return false; }

// Define the load_weights function
void load_weights(const char *filepath, float *buffer, size_t num_elements) {
  FILE *file = fopen(filepath, "rb");
  if (!file) {
    perror("Error opening file");
    exit(EXIT_FAILURE);
  }

  size_t read_count = fread(buffer, sizeof(float), num_elements, file);
  if (read_count != num_elements) {
    perror("Error reading file");
    fclose(file);
    exit(EXIT_FAILURE);
  }

  fclose(file);
}
// Define the load_int_array function
void load_int_array(const char *filename, int *data, size_t size) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    perror("Error opening file");
    return;
  }

  size_t read_count = fread(data, sizeof(int), size, file);
  if (read_count != size) {
    fprintf(stderr, "Error: Could not read all data from file %s\n", filename);
  }

  fclose(file);
}
// Helper function to print an array of floats
void print_float_array(float *array, size_t size) {
  for (size_t i = 0; i < size; i++) {
    printf("%.6f ", array[i]);
  }
  printf("\n");
}

void layernorm_forward_cpu(float *out, const float *inp, const float *weight,
                           const float *bias, int B, int T, int C) {
  float eps = 1e-5f;
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the input position inp[b,t,:]
      const float *x = inp + b * T * C + t * C;
      // calculate the mean
      float m = 0.0f;
      for (int i = 0; i < C; i++) {
        m += x[i];
      }
      m = m / C;
      // calculate the variance (without any bias correction)
      float v = 0.0f;
      for (int i = 0; i < C; i++) {
        float xshift = x[i] - m;
        v += xshift * xshift;
      }
      v = v / C;
      // calculate the rstd
      float s = 1.0f / sqrtf(v + eps);
      // seek to the output position in out[b,t,:]
      float *out_bt = out + b * T * C + t * C;
      for (int i = 0; i < C; i++) {
        float n = (s * (x[i] - m));        // normalized output
        float o = n * weight[i] + bias[i]; // scale and shift it
        out_bt[i] = o;                     // write
      }
    }
  }
}

bool is_training();
BertConfig default_config();
BertEmbeddings init_bert_embeddings(BertConfig *config);
void bert_word_embeddings_forward(BertEmbeddings *embeddings, int *input_ids,
                                  int batch_size, int seq_length,
                                  float *output);
void load_weights(const char *filename, float *weights, size_t num_weights);
void load_int_array(const char *filename, int *array, size_t num_elements);

#endif // BERT_H
