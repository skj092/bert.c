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
  int vocab_size;
  int hidden_size;
  int max_position_embeddings;
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
  config.vocab_size = 30522;
  config.hidden_size = 768;
  config.max_position_embeddings = 512;
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
void load_int_array(const char* filename, int* data, size_t size) {
    FILE* file = fopen(filename, "rb");
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

bool is_training();
BertConfig default_config();
BertEmbeddings init_bert_embeddings(BertConfig *config);
void bert_word_embeddings_forward(BertEmbeddings *embeddings, int *input_ids,
                                  int batch_size, int seq_length,
                                  float *output);
void load_weights(const char *filename, float *weights, size_t num_weights);
void load_int_array(const char *filename, int *array, size_t num_elements);

#endif // BERT_H
