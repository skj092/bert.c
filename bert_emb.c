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
  load_weights("./bins/weights/embeddings_position_embeddings_weight.bin",
               embeddings.position_embeddings, pos_emb_size / sizeof(float));
  load_weights("./bins/weights/embeddings_token_type_embeddings_weight.bin",
               embeddings.token_type_embeddings, type_emb_size / sizeof(float));

  // Initialize layer norm weights to 1 and biases to 0
  for (int i = 0; i < config->hidden_size; i++) {
    embeddings.layer_norm_weight[i] = 1.0f;
    embeddings.layer_norm_bias[i] = 0.0f;
  }

  return embeddings;
}

float *bert_embeddings_forward(BertEmbeddings *embeddings, int *input_ids,
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
  // (bs, sl, 768)
  return temp_embeddings;
}

// Kahan summation for more accurate floating-point summation
double kahan_sum(const float *arr, int n) {
  double sum = 0.0;
  double c = 0.0; // A running compensation for lost low-order bits
  for (int i = 0; i < n; i++) {
    double y = (double)arr[i] - c;
    double t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}

void layernorm_forward(float *out, float *inp, float *weight, float *bias,
                       int B, int T, int C) {
  // Use more precise epsilon
  const double eps = 1e-12;

  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // Pointer to current input slice
      float *x = inp + b * T * C + t * C;

      // Compute mean using Kahan summation
      double mean = kahan_sum(x, C) / C;

      // Compute variance
      double variance = 0.0;
      double c = 0.0;
      for (int i = 0; i < C; i++) {
        double xshift = (double)x[i] - mean;
        double y = xshift * xshift - c;
        double t = variance + y;
        c = (t - variance) - y;
        variance = t;
      }
      variance /= C;

      // Compute reciprocal standard deviation
      double rstd = 1.0 / sqrt(variance + eps);

      // Pointer to current output slice
      float *out_bt = out + b * T * C + t * C;

      // Normalize, scale, and shift
      for (int i = 0; i < C; i++) {
        // Compute normalized value using high precision
        double normalized = ((double)x[i] - mean) * rstd;

        // Scale and shift
        out_bt[i] = (float)(normalized * weight[i] + bias[i]);
      }
    }
  }
}

void dropout_forward(float *output, const float *input, float dropout_prob,
                     int batch_size, int seq_length, int hidden_size,
                     int training) {
  int total_elements = batch_size * seq_length * hidden_size;

  // In inference mode, just copy the input
  if (!training) {
    memcpy(output, input, total_elements * sizeof(float));
    return;
  }

  // In training mode, randomly zero out elements
  float scale = 1.0f / (1.0f - dropout_prob);

  // Set random seed for reproducibility in testing
  srand(42); // Use a fixed seed for testing

  for (int i = 0; i < total_elements; i++) {
    float r = (float)rand() / RAND_MAX;
    if (r < dropout_prob) {
      output[i] = 0.0f;
    } else {
      output[i] =
          input[i] * scale; // Scale by 1/(1-p) to maintain expected value
    }
  }
}
