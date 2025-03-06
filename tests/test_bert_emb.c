#include "../bert.h"
#include "../bert_emb.c" // Include the original file to reuse its definitions
#include <stdio.h>
#include <stdlib.h>

float *load_binary_file(const char *filename, size_t *num_elements) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Error: Cannot open file %s\n", filename);
    *num_elements = 0;
    return NULL;
  }

  // Determine file size
  fseek(file, 0, SEEK_END);
  long file_size = ftell(file);
  rewind(file);

  // Calculate number of elements
  *num_elements = file_size / sizeof(float);

  // Allocate memory
  float *data = (float *)malloc(file_size);
  if (!data) {
    fprintf(stderr, "Error: Memory allocation failed for %s\n", filename);
    fclose(file);
    *num_elements = 0;
    return NULL;
  }

  // Read file contents
  size_t elements_read = fread(data, sizeof(float), *num_elements, file);
  fclose(file);

  if (elements_read != *num_elements) {
    fprintf(stderr, "Error: Could not read entire file %s\n", filename);
    free(data);
    *num_elements = 0;
    return NULL;
  }

  return data;
}

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
  float *x = NULL;       // input
  float *w = NULL;       // weights
  float *b = NULL;       // biases
  float *ref_out = NULL; // reference output
  size_t elements;

  BertConfig config = default_config();
  // Initialize embeddings
  BertEmbeddings embeddings = init_bert_embeddings(&config);

  // Load input IDs
  int batch_size = 2;
  int seq_length = 128;
  int input_ids[batch_size * seq_length];
  load_int_array("bins/input_ids.bin", input_ids, batch_size * seq_length);
  print_int_array(input_ids, 10);
  //
  // // nitialize token id
  int token_id[batch_size * seq_length];
  load_int_array("bins/token_type_ids.bin", token_id, batch_size * seq_length);
  //
  // // Perform forward pass
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

  x = load_binary_file("bins/ln_i.bin", &elements);
  w = load_binary_file("bins/ln_w.bin", &elements);
  b = load_binary_file("bins/ln_b.bin", &elements);
  ref_out = load_binary_file("bins/ln_o.bin", &elements);

  // Perform LayerNorm forward pass
  float *c_out = (float *)malloc(batch_size * seq_length * 768 * sizeof(float));

  layernorm_forward(c_out, combined_emb, w, b, batch_size, seq_length, 768);
  check_tensor(c_out, ref_out, seq_length * batch_size * config.hidden_size,
               "ln_out-val");

  // dropout
  float *ln_output = c_out;  // Reuse the output from layer norm test
  float *ref_dropout_out = NULL;
  float *dropout_output = (float *)malloc(batch_size * seq_length * config.hidden_size * sizeof(float));
  ref_dropout_out = load_binary_file("bins/dropout_output.bin", &elements);
  dropout_forward(dropout_output, ln_output, config.hidden_dropout_prob,
                batch_size, seq_length, config.hidden_size, 0);
  check_tensor(dropout_output, ref_dropout_out,
             batch_size * seq_length * config.hidden_size,
             "dropout-inference-mode");


  free(w);
  free(b);
  free(ref_out);
  free(c_out);
  free(dropout_output);
  free(ref_dropout_out);

  return 0;
}
