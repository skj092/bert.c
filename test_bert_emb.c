#include "bert.h"
#include "bert_emb.c" // Include the original file to reuse its definitions
#include <stdio.h>
#include <stdlib.h>

// poor man's tensor checker
int check_tensor(float *a, float *b, int n, const char *label) {
  int print_upto = 4;
  int ok = 1;
  float maxdiff = 0.0f;
  float tol = 2e-2f;
  printf("%s\n", label);
  for (int i = 0; i < n; i++) {
    // look at the diffence at position i of these two tensors
    float diff = fabsf(a[i] - b[i]);

    // keep track of the overall error
    ok = ok && (diff <= tol);
    if (diff > maxdiff) {
      maxdiff = diff;
    }

    // for the first few elements of each tensor, pretty print
    // the actual numbers, so we can do a visual, qualitative proof/assessment
    if (i < print_upto) {
      if (diff <= tol) {
        if (i < print_upto) {
          printf("OK ");
        }
      } else {
        if (i < print_upto) {
          printf("NOT OK ");
        }
      }
      printf("%f %f\n", a[i], b[i]);
    }
  }
  // print the final result for this tensor
  if (ok) {
    printf("TENSOR OK, maxdiff = %e\n", maxdiff);
  } else {
    printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
  }
  return ok;
}

// First, modify your main function to add debug info:
int main() {
  // Initialize configuration
  BertConfig config = default_config();
  // Initialize embeddings
  BertEmbeddings embeddings = init_bert_embeddings(&config);

  // verify the word embedding weights
  float *hf_we_weight;
  hf_we_weight =
      (float *)malloc(config.vocab_size * config.hidden_size * sizeof(float));
  load_weights("bins/bert_word_embeddings.bin", hf_we_weight,
               config.vocab_size * config.hidden_size);
  check_tensor(embeddings.word_embeddings, hf_we_weight,
               config.hidden_size * config.vocab_size, "we-weight");

  // Load input IDs
  int batch_size = 2;
  int seq_length = 128;
  int input_ids[batch_size * seq_length];
  load_int_array("bins/input_ids.bin", input_ids, batch_size * seq_length);

  // Perform forward pass
  float *output = (float *)malloc(batch_size * seq_length * config.hidden_size *
                                  sizeof(float));
  // Clear output to ensure we don't have uninitialized memory
  memset(output, 0,
         batch_size * seq_length * config.hidden_size * sizeof(float));

  // Run forward pass
  bert_word_embeddings_forward(&embeddings, input_ids, batch_size, seq_length,
                               output);

  // load hf model output
  float *we_output_hf = (float *)malloc(batch_size * seq_length *
                                        config.hidden_size * sizeof(float));
  load_weights("bins/word_embeddings_output.bin", we_output_hf,
               batch_size * seq_length * config.hidden_size);

  // verify full tensor
  check_tensor(output, we_output_hf,
               batch_size * seq_length * config.hidden_size, "we_output");

  // Free allocated memory
  free(embeddings.word_embeddings);
  // Remove this line to fix segfault - position_embeddings was never allocated
  // free(embeddings.position_embeddings);
  free(output);
  free(we_output_hf);
  free(hf_we_weight);

  return 0;
}
