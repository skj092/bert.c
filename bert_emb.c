#include "bert.h"

// Define the init_bert_embeddings function
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

  // Load pretrained weights
  load_weights("bins/weights/embeddings_word_embeddings_weight.bin",
               embeddings.word_embeddings, word_emb_size / sizeof(float));
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

float* bert_embeddings_forward(BertEmbeddings *embeddings, int *input_ids,
                             int *token_type_ids, int batch_size,
                             int seq_length, float *output) {
  int hidden_size = embeddings->config->hidden_size;

  // Initialize output with zeros
  size_t output_size = batch_size * seq_length * hidden_size * sizeof(float);
  memset(output, 0, output_size);

  // Create temporary buffer for embeddings
  float *temp_embeddings = (float *)malloc(output_size);
  memset(temp_embeddings, 0, output_size);

  // Process each item in batch
  for (int b = 0; b < batch_size; b++) {
    for (int s = 0; s < seq_length; s++) {
      int idx = b * seq_length + s;
      int word_idx = input_ids[idx];
      int type_idx = token_type_ids ? token_type_ids[idx] : 0;

      // Get word embedding
      for (int h = 0; h < hidden_size; h++) {
        int emb_idx = b * seq_length * hidden_size + s * hidden_size + h;
        int word_emb_idx = word_idx * hidden_size + h;
        temp_embeddings[emb_idx] = embeddings->word_embeddings[word_emb_idx];

        // Add position embedding
        int pos_emb_idx = s * hidden_size + h;
        temp_embeddings[emb_idx] +=
            embeddings->position_embeddings[pos_emb_idx];

        // Add token type embedding
        int type_emb_idx = type_idx * hidden_size + h;
        temp_embeddings[emb_idx] +=
            embeddings->token_type_embeddings[type_emb_idx];
      }
    }
  }
  return temp_embeddings;
}

// // Define the bert_word_embeddings_forward function
// void bert_word_embeddings_forward(BertEmbeddings *embeddings, int *input_ids,
//                                   int batch_size, int seq_length,
//                                   float *output) {
//   int hidden_size = embeddings->config->hidden_size;
//
//   // Directly extract embeddings for each input ID
//   int total_tokens = batch_size * seq_length;
//   for (int i = 0; i < total_tokens; i++) {
//     int word_idx = input_ids[i];
//
//     // Bounds check
//     if (word_idx < 0 || word_idx >= embeddings->config->vocab_size) {
//       fprintf(stderr, "Error: Invalid word index %d\n", word_idx);
//       exit(EXIT_FAILURE);
//     }
//
//     // Copy embedding directly to output
//     memcpy(&output[i * hidden_size],
//            &embeddings->word_embeddings[word_idx * hidden_size],
//            hidden_size * sizeof(float));
//   }
// }
