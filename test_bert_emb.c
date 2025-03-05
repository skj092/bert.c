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

  // Load input IDs
  int batch_size = 2;
  int seq_length = 128;
  int input_ids[batch_size * seq_length];
  load_int_array("bins/input_ids.bin", input_ids, batch_size * seq_length);
  print_int_array(input_ids, 10);

  // nitialize token id
  int token_id[batch_size * seq_length];
  load_int_array("bins/token_type_ids.bin", token_id, batch_size * seq_length);

  // Perform forward pass
  float *output = (float *)malloc(batch_size * seq_length * config.hidden_size *
                                  sizeof(float));
  float *combined_emb = (float *)malloc(batch_size * seq_length *
                                        config.hidden_size * sizeof(float));
  load_weights("bins/combined_embeddings.bin", combined_emb,
               seq_length * config.hidden_size * batch_size);

  output = bert_embeddings_forward(&embeddings, input_ids, token_id, batch_size,
                                   seq_length, output);
  check_tensor(output, combined_emb,
               seq_length * batch_size * config.hidden_size,
               "combined_emb-val");

  // layer norm
  // float *ln_output = (float *)malloc(batch_size * seq_length *
  // config.hidden_size * sizeof(float)); float *gamma = (float
  // *)malloc(config.hidden_size * sizeof(float)); float *beta = (float
  // *)malloc(config.hidden_size * sizeof(float));
  // load_weights("./bins/weights/embeddings_LayerNorm_weight.bin", gamma,
  // config.hidden_size);
  // load_weights("./bins/weights/embeddings_LayerNorm_bias.bin", beta,
  // config.hidden_size); layernorm_forward_cpu(ln_output, combined_emb, gamma,
  // beta, batch_size, seq_length, config.hidden_size); float
  // *expected_ln_output = (float *)malloc(batch_size * seq_length *
  // config.hidden_size * sizeof(float));
  // load_weights("bins/layernorm_output.bin", expected_ln_output, batch_size *
  // seq_length * config.hidden_size); check_tensor(ln_output,
  // expected_ln_output, batch_size * seq_length * config.hidden_size,
  // "layernorm_output");

  // free(ln_output);
  // free(expected_ln_output);
  // free(gamma);
  // free(beta);

  return 0;
}
