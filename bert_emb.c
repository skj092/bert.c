#include "utils.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define implementation for is_training() from bert_utils.h
bool is_training() {
  // This would be controlled by a global flag or parameter in a real
  // implementation
  return false;
}

// Initialize BERT embeddings
BertEmbeddings init_bert_embeddings(BertConfig *config) {
  BertEmbeddings embeddings;
  embeddings.config = config;

  // Allocate memory for embeddings
  size_t word_emb_size =
      config->vocab_size * config->hidden_size * sizeof(float);
  size_t pos_emb_size =
      config->max_position_embeddings * config->hidden_size * sizeof(float);
  size_t type_emb_size =
      config->type_vocab_size * config->hidden_size * sizeof(float);
  size_t norm_size = config->hidden_size * sizeof(float);

  embeddings.word_embeddings = (float *)malloc(word_emb_size);
  embeddings.position_embeddings = (float *)malloc(pos_emb_size);
  embeddings.token_type_embeddings = (float *)malloc(type_emb_size);
  embeddings.layer_norm_weight = (float *)malloc(norm_size);
  embeddings.layer_norm_bias = (float *)malloc(norm_size);

  // Initialize with zeros (in practice, these would be loaded from a
  // pre-trained model)
  memset(embeddings.word_embeddings, 0, word_emb_size);
  memset(embeddings.position_embeddings, 0, pos_emb_size);
  memset(embeddings.token_type_embeddings, 0, type_emb_size);
  //
  // // Load pretrained weights
  load_weights("bins/bert_word_embeddings.bin", embeddings.word_embeddings,
               word_emb_size / sizeof(float));
  load_weights("bins/bert_position_embeddings.bin",
               embeddings.position_embeddings, pos_emb_size / sizeof(float));
  load_weights("bins/bert_token_type_embeddings.bin",
               embeddings.token_type_embeddings, type_emb_size / sizeof(float));

  // Initialize layer norm weights to 1 and biases to 0
  for (int i = 0; i < config->hidden_size; i++) {
    embeddings.layer_norm_weight[i] = 1.0f;
    embeddings.layer_norm_bias[i] = 0.0f;
  }

  return embeddings;
}

void bert_word_embeddings_forward(BertEmbeddings *embeddings, int *input_ids,
                                  int batch_size, int seq_length,
                                  float *output) {
  int hidden_size = embeddings->config->hidden_size;

  // Initialize output with zeros
  size_t output_size = batch_size * seq_length * hidden_size * sizeof(float);
  memset(output, 0, output_size);

  // Process each item in batch
  for (int b = 0; b < batch_size; b++) {
    for (int s = 0; s < seq_length; s++) {
      int idx = b * seq_length + s;
      int word_idx = input_ids[idx];

      if (word_idx < 0 || word_idx >= embeddings->config->vocab_size) {
        fprintf(stderr, "Error: Invalid word index %d\n", word_idx);
        exit(EXIT_FAILURE);
      }

      // Get word embedding
      for (int h = 0; h < hidden_size; h++) {
        // Calculate proper 3D index for output: [batch, sequence, hidden]
        int emb_idx = (b * seq_length * hidden_size) + (s * hidden_size) + h;
        int word_emb_idx = word_idx * hidden_size + h;
        output[emb_idx] = embeddings->word_embeddings[word_emb_idx];

        // Debug print: Print the first few values for the first token
        if (b == 0 && s == 0 && h < 16) {
          printf("Token 6054 at position [0,0], dim %d: %.6f\n", h,
                 output[emb_idx]);
        }
      }
    }
  }

  // Add debugging to check the second token's embedding
  printf("\nValues for second token (6653) at position [0,1]:\n");
  for (int h = 0; h < 8; h++) {
    int second_token_idx = hidden_size + h; // First batch, second token
    int word_emb_idx =
        input_ids[1] * hidden_size + h; // Embedding for second token
    printf("dim %d: %.6f (from embedding weight: %.6f)\n", h,
           output[second_token_idx], embeddings->word_embeddings[word_emb_idx]);
  }
}
int main() {
  int batch_size = 2;
  int seq_length = 128;
  float *output;

  BertConfig config = default_config();
  BertEmbeddings embeddings = init_bert_embeddings(&config);

  printf("First few word embedding values:\n");
  print_weights(embeddings.word_embeddings, 16);

  // Print embedding values for token ID 6054 (first token in your sequence)
  printf("\nEmbedding values for token ID 6054:\n");
  int token_6054_offset = 6054 * config.hidden_size;
  print_weights(&embeddings.word_embeddings[token_6054_offset], 16);

  int input_ids[batch_size * seq_length];
  load_int_array("bins/input_ids.bin", input_ids, batch_size * seq_length);

  printf("\nFirst 16 input IDs:\n");
  print_int_array(input_ids, 16);

  int token_type_ids[batch_size * seq_length];
  // Fix the loop bounds to avoid out-of-bounds access
  for (int bs = 0; bs < batch_size; bs++) {
    for (int sl = 0; sl < seq_length; sl++) {
      token_type_ids[bs * seq_length + sl] = 0;
    }
  }

  output = (float *)malloc(batch_size * seq_length * config.hidden_size *
                           sizeof(float));
  bert_word_embeddings_forward(&embeddings, input_ids, batch_size, seq_length,
                               output);

  printf("\nWord embedding output (first 16 values):\n");
  print_weights(output, 16);

  // Print values for the first few tokens in different positions
  printf("\nValues for token at position [0,1] (token ID: %d):\n",
         input_ids[1]);
  print_weights(&output[config.hidden_size], 8); // First batch, second position

  printf("\nValues for token at position [1,0] (token ID: %d):\n",
         input_ids[seq_length]);
  print_weights(&output[seq_length * config.hidden_size],
                8); // Second batch, first position

  return 0;
}
