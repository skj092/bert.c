#include "../attention/utils.c"
#include "../encoder/bert_utils.c"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Self-attention component
typedef struct {
  float *query_weight; // [hidden_size, hidden_size]
  float *query_bias;   // [hidden_size]
  float *key_weight;   // [hidden_size, hidden_size]
  float *key_bias;     // [hidden_size]
  float *value_weight; // [hidden_size, hidden_size]
  float *value_bias;   // [hidden_size]
  int num_heads;
  int head_dim;
  float scale;
} BertSelfAttention;

// Self-output component for attention
typedef struct {
  float *dense_weight;      // [hidden_size, hidden_size]
  float *dense_bias;        // [hidden_size]
  float *layer_norm_weight; // [hidden_size]
  float *layer_norm_bias;   // [hidden_size]
  float dropout_prob;
  float layer_norm_eps;
} BertSelfOutput;

// Attention component
typedef struct {
  BertSelfAttention self;
  BertSelfOutput output;
} BertAttention;

// Intermediate component
typedef struct {
  float *dense_weight; // [hidden_size, intermediate_size]
  float *dense_bias;   // [intermediate_size]
} BertIntermediate;

// Output component
typedef struct {
  float *dense_weight;      // [intermediate_size, hidden_size]
  float *dense_bias;        // [hidden_size]
  float *layer_norm_weight; // [hidden_size]
  float *layer_norm_bias;   // [hidden_size]
  float dropout_prob;
  float layer_norm_eps;
} BertOutput;

// Layer component
typedef struct {
  BertAttention attention;
  BertIntermediate intermediate;
  BertOutput output;
} BertLayer;

void load_output_checkpoint(BertOutput *bo, BertConfig config) {
  const char *paths[] = {
      "bins/weights/encoder_layer_0_output_dense_weight.bin",
      "bins/weights/encoder_layer_0_output_dense_bias.bin",
      "bins/weights/encoder_layer_0_output_LayerNorm_weight.bin",
      "bins/weights/encoder_layer_0_output_LayerNorm_bias.bin",
  };

  printf("Loading model weights and biases...\n");
  int total_size;

  bo->dense_weight = load_tensor(paths[0], &total_size);
  if (!bo->dense_weight) {
    fprintf(stderr, "Error loading dense_weight\n");
    return;
  }

  bo->dense_bias = load_tensor(paths[1], &total_size);
  if (!bo->dense_bias) {
    fprintf(stderr, "Error loading dense_bias\n");
    free(bo->dense_weight);
    return;
  }

  bo->layer_norm_weight = load_tensor(paths[2], &total_size);
  if (!bo->layer_norm_weight) {
    fprintf(stderr, "Error loading layer_norm_weight\n");
    free(bo->dense_weight);
    free(bo->dense_bias);
    return;
  }

  bo->layer_norm_bias = load_tensor(paths[3], &total_size);
  if (!bo->layer_norm_bias) {
    fprintf(stderr, "Error loading layer_norm_bias\n");
    free(bo->dense_weight);
    free(bo->dense_bias);
    free(bo->layer_norm_weight);
    return;
  }

  bo->layer_norm_eps = config.layer_norm_eps;
  bo->dropout_prob = config.hidden_dropout_prob;
}
void load_intermediate_checkpoint(BertIntermediate *intermediate,
                                  BertConfig config) {
  const char *paths[] = {
      "bins/weights/encoder_layer_0_intermediate_dense_weight.bin",
      "bins/weights/encoder_layer_0_intermediate_dense_bias.bin",
  };

  printf("Loading model weights and biases...\n");
  int total_size;

  intermediate->dense_weight = load_tensor(paths[0], &total_size);
  if (!intermediate->dense_weight) {
    fprintf(stderr, "Error loading dense_weight\n");
    return;
  }

  intermediate->dense_bias = load_tensor(paths[1], &total_size);
  if (!intermediate->dense_bias) {
    fprintf(stderr, "Error loading dense_bias\n");
    free(intermediate->dense_weight);
    return;
  }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float *out, float *inp, int N) {
  // (approximate) GeLU elementwise non-linearity in the MLP block of
  // Transformer
  for (int i = 0; i < N; i++) {
    float x = inp[i];
    float cube = 0.044715f * x * x * x;
    out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
  }
}

void intermediate_forward(BertIntermediate intermediate, float *input,
                          float *output) {
  matmul_forward(output, input, intermediate.dense_weight,
                 intermediate.dense_bias, 2, 128, 768, 768);
  int total_size;
  float *h = load_tensor("bins/layer0_intermediate_dense.bin", &total_size);
  if (h) {
    check_tensor(h, output, 5, "intermediate dense output");
    free(h);
  }
  float *l_fch_gelu = malloc(2 * 128 * 768 * 4 * sizeof(float));
  gelu_forward(l_fch_gelu, output, 2 * 128 * 768 * 4);
  print_float_array(l_fch_gelu, 5);
  float *g =
      load_tensor("bins/layer0_intermediate_activation.bin", &total_size);
  if (g) {
    check_tensor(g, l_fch_gelu, 5, "intermediate activation output");
    free(g);
  }
}

void load_from_checkpoint1(BertSelfOutput *bso, BertConfig config) {
  const char *paths[] = {
      "bins/weights/encoder_layer_0_attention_output_dense_weight.bin",
      "bins/weights/encoder_layer_0_attention_output_dense_bias.bin",
      "bins/weights/encoder_layer_0_attention_output_LayerNorm_weight.bin",
      "bins/weights/encoder_layer_0_attention_output_LayerNorm_bias.bin"};

  printf("Loading model weights and biases...\n");
  int total_size;

  bso->dense_weight = load_tensor(paths[0], &total_size);
  if (!bso->dense_weight) {
    fprintf(stderr, "Error loading dense_weight\n");
    return;
  }

  bso->dense_bias = load_tensor(paths[1], &total_size);
  if (!bso->dense_bias) {
    fprintf(stderr, "Error loading dense_bias\n");
    free(bso->dense_weight);
    return;
  }

  bso->layer_norm_weight = load_tensor(paths[2], &total_size);
  if (!bso->layer_norm_weight) {
    fprintf(stderr, "Error loading layer_norm_weight\n");
    free(bso->dense_weight);
    free(bso->dense_bias);
    return;
  }

  bso->layer_norm_bias = load_tensor(paths[3], &total_size);
  if (!bso->layer_norm_bias) {
    fprintf(stderr, "Error loading layer_norm_bias\n");
    free(bso->dense_weight);
    free(bso->dense_bias);
    free(bso->layer_norm_weight);
    return;
  }

  bso->layer_norm_eps = config.layer_norm_eps;
  bso->dropout_prob = config.hidden_dropout_prob;
}

void load_from_checkpoint(BertSelfAttention *bsa, BertConfig config) {
  const char *paths[] = {"bins/tmp/qw.bin", "bins/tmp/kw.bin",
                         "bins/tmp/vw.bin", "bins/tmp/qb.bin",
                         "bins/tmp/kb.bin", "bins/tmp/vb.bin"};

  int total_size;

  bsa->query_weight = load_tensor(paths[0], &total_size);
  if (!bsa->query_weight) {
    fprintf(stderr, "Error loading query_weight\n");
    return;
  }

  bsa->key_weight = load_tensor(paths[1], &total_size);
  if (!bsa->key_weight) {
    fprintf(stderr, "Error loading key_weight\n");
    free(bsa->query_weight);
    return;
  }

  bsa->value_weight = load_tensor(paths[2], &total_size);
  if (!bsa->value_weight) {
    fprintf(stderr, "Error loading value_weight\n");
    free(bsa->query_weight);
    free(bsa->key_weight);
    return;
  }

  bsa->query_bias = load_tensor(paths[3], &total_size);
  if (!bsa->query_bias) {
    fprintf(stderr, "Error loading query_bias\n");
    free(bsa->query_weight);
    free(bsa->key_weight);
    free(bsa->value_weight);
    return;
  }

  bsa->key_bias = load_tensor(paths[4], &total_size);
  if (!bsa->key_bias) {
    fprintf(stderr, "Error loading key_bias\n");
    free(bsa->query_weight);
    free(bsa->key_weight);
    free(bsa->value_weight);
    free(bsa->query_bias);
    return;
  }

  bsa->value_bias = load_tensor(paths[5], &total_size);
  if (!bsa->value_bias) {
    fprintf(stderr, "Error loading value_bias\n");
    free(bsa->query_weight);
    free(bsa->key_weight);
    free(bsa->value_weight);
    free(bsa->query_bias);
    free(bsa->key_bias);
    return;
  }

  bsa->num_heads = config.num_attention_heads;
  bsa->head_dim = config.hidden_size / config.num_attention_heads;
  bsa->scale = 1.0f / sqrtf(bsa->head_dim);
}

void bert_attention_forward(BertSelfAttention bsa, float *input_tensor,
                            float *attn_output) {
  int batch_size = 2;
  int seq_length = 128;
  int hidden_size = 768;
  int num_heads = bsa.num_heads;
  int head_dim = bsa.head_dim;
  const int total_seq_hidden = batch_size * seq_length * hidden_size;

  float *q_proj = malloc(total_seq_hidden * sizeof(float));
  float *k_proj = malloc(total_seq_hidden * sizeof(float));
  float *v_proj = malloc(total_seq_hidden * sizeof(float));
  if (!q_proj || !k_proj || !v_proj) {
    fprintf(stderr, "Memory allocation failed for projections\n");
    free(q_proj);
    free(k_proj);
    free(v_proj);
    return;
  }

  linear_projection(input_tensor, bsa.query_weight, bsa.query_bias, q_proj,
                    batch_size, seq_length, hidden_size, hidden_size);
  linear_projection(input_tensor, bsa.key_weight, bsa.key_bias, k_proj,
                    batch_size, seq_length, hidden_size, hidden_size);
  linear_projection(input_tensor, bsa.value_weight, bsa.value_bias, v_proj,
                    batch_size, seq_length, hidden_size, hidden_size);

  const int total_head_seq_dim = batch_size * num_heads * seq_length * head_dim;
  float *q_reshaped = malloc(total_head_seq_dim * sizeof(float));
  float *k_reshaped = malloc(total_head_seq_dim * sizeof(float));
  float *v_reshaped = malloc(total_head_seq_dim * sizeof(float));
  if (!q_reshaped || !k_reshaped || !v_reshaped) {
    fprintf(stderr, "Memory allocation failed for reshaped tensors\n");
    goto cleanup_reshape;
  }

  reshape_for_multihead(q_proj, q_reshaped, batch_size, seq_length, num_heads,
                        head_dim);
  reshape_for_multihead(k_proj, k_reshaped, batch_size, seq_length, num_heads,
                        head_dim);
  reshape_for_multihead(v_proj, v_reshaped, batch_size, seq_length, num_heads,
                        head_dim);

  const int attn_size = batch_size * num_heads * seq_length * seq_length;
  float *attn_scores = malloc(attn_size * sizeof(float));
  float *attn_scaled = malloc(attn_size * sizeof(float));
  float *attn_softmax = malloc(attn_size * sizeof(float));
  if (!attn_scores || !attn_scaled || !attn_softmax) {
    fprintf(stderr, "Memory allocation failed for attention tensors\n");
    goto cleanup_attention;
  }

  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < num_heads; h++) {
      float *q_ptr = q_reshaped + (b * num_heads + h) * seq_length * head_dim;
      float *k_ptr = k_reshaped + (b * num_heads + h) * seq_length * head_dim;
      float *v_ptr = v_reshaped + (b * num_heads + h) * seq_length * head_dim;
      float *scores_ptr =
          attn_scores + (b * num_heads + h) * seq_length * seq_length;
      float *scaled_ptr =
          attn_scaled + (b * num_heads + h) * seq_length * seq_length;
      float *softmax_ptr =
          attn_softmax + (b * num_heads + h) * seq_length * seq_length;

      matmul_transpose(q_ptr, k_ptr, scores_ptr, seq_length, head_dim,
                       seq_length);
      for (int i = 0; i < seq_length * seq_length; i++) {
        scaled_ptr[i] = scores_ptr[i] * bsa.scale;
      }
      softmax(scaled_ptr, softmax_ptr, seq_length, seq_length);
      matmul(softmax_ptr, v_ptr,
             attn_output + (b * num_heads + h) * seq_length * head_dim,
             seq_length, seq_length, head_dim);
    }
  }

  float *context = malloc(total_seq_hidden * sizeof(float));
  if (!context) {
    fprintf(stderr, "Memory allocation failed for context\n");
    goto cleanup_attention;
  }

  reshape_from_multihead(attn_output, context, batch_size, seq_length,
                         num_heads, head_dim);
  memcpy(attn_output, context, total_seq_hidden * sizeof(float));

cleanup_attention:
  free(attn_softmax);
  free(attn_scaled);
  free(attn_scores);
cleanup_reshape:
  free(v_reshaped);
  free(k_reshaped);
  free(q_reshaped);
  free(v_proj);
  free(k_proj);
  free(q_proj);
  free(context);
}
void bert_output_forward(BertOutput bo, float *output, float *hidden_states,
                         float *input_tensor, int B, int T, int C) {
  float *temp_buffer = malloc(B * T * C * sizeof(float));
  if (!temp_buffer) {
    fprintf(stderr, "Error: Memory allocation failed for temp buffer\n");
    return;
  }
  printf("bertoutput\n");

  matmul_forward(temp_buffer, hidden_states, bo.dense_weight, bo.dense_bias, B,
                 T, C, C);
  print_float_array(temp_buffer, 5);
  apply_dropout(temp_buffer, temp_buffer, bo.dropout_prob, B * T * C);

  float *residual = malloc(B * T * C * sizeof(float));
  if (!residual) {
    fprintf(stderr, "Error: Memory allocation failed for residual\n");
    free(temp_buffer);
    return;
  }

  add_tensors(residual, temp_buffer, input_tensor, B * T * C);
  layernorm_forward_cpu(output, residual, bo.layer_norm_weight,
                        bo.layer_norm_bias, B, T, C, bo.layer_norm_eps);

  float *l_fch_gelu = malloc(2 * 128 * 768 * sizeof(float));
  int total_size;
  // print_float_array(l_fch_gelu, 5);
  float *oo = load_tensor("bins/layer0_output_layernorm.bin", &total_size);
  if (oo) {
    check_tensor(oo, output, 5, "intermediate activation output");
    free(oo);
    free(temp_buffer);
    free(residual);
  }
}

void bert_selfoutput_forward(BertSelfOutput bso, float *output,
                             float *hidden_states, float *input_tensor, int B,
                             int T, int C) {
  float *temp_buffer = malloc(B * T * C * sizeof(float));
  if (!temp_buffer) {
    fprintf(stderr, "Error: Memory allocation failed for temp buffer\n");
    return;
  }

  matmul_forward(temp_buffer, hidden_states, bso.dense_weight, bso.dense_bias,
                 B, T, C, C);
  apply_dropout(temp_buffer, temp_buffer, bso.dropout_prob, B * T * C);

  float *residual = malloc(B * T * C * sizeof(float));
  if (!residual) {
    fprintf(stderr, "Error: Memory allocation failed for residual\n");
    free(temp_buffer);
    return;
  }

  add_tensors(residual, temp_buffer, input_tensor, B * T * C);
  layernorm_forward_cpu(output, residual, bso.layer_norm_weight,
                        bso.layer_norm_bias, B, T, C, bso.layer_norm_eps);

  free(temp_buffer);
  free(residual);
}

void attention_forward(BertAttention attention, BertConfig config, float *input,
                       float *output) {
  int total_size = 2 * 128 * 768;

  bert_attention_forward(attention.self, input, output);

  float *h = load_tensor("bins/t1.bin", &total_size);
  if (h) {
    check_tensor(h, output, 5, "self attention_output");
    free(h);
  }

  bert_selfoutput_forward(attention.output, output, output, input, 2, 128, 768);

  float *expected_out =
      load_tensor("bins/layer0_self_output_layernorm.bin", &total_size);
  if (expected_out) {
    check_tensor(output, expected_out, 5, "attention_output");
    free(expected_out);
  }
}

int main() {
  BertConfig config = default_config();
  const int total_seq_hidden = 2 * 128 * 768;

  // ===============Initialize and load module=================
  //
  BertSelfAttention self_attention;
  load_from_checkpoint(&self_attention, config);
  if (!self_attention.query_weight || !self_attention.key_weight ||
      !self_attention.value_weight || !self_attention.query_bias ||
      !self_attention.key_bias || !self_attention.value_bias) {
    fprintf(stderr, "Failed to load self attention weights\n");
    goto cleanup_self_attention;
  }

  BertSelfOutput self_output;
  load_from_checkpoint1(&self_output, config);
  if (!self_output.dense_weight || !self_output.dense_bias ||
      !self_output.layer_norm_weight || !self_output.layer_norm_bias) {
    fprintf(stderr, "Failed to load self output weights\n");
    goto cleanup_self_output;
  }

  BertIntermediate intermediate;
  load_intermediate_checkpoint(&intermediate, config);
  if (!intermediate.dense_weight || !self_output.dense_bias) {
    fprintf(stderr, "Failed to load self output weights\n");
    goto cleanup_intermediate;
  }

  BertOutput bertoutput;
  load_output_checkpoint(&bertoutput, config);
  if (!bertoutput.dense_weight || !bertoutput.dense_bias ||
      !bertoutput.layer_norm_weight || !bertoutput.layer_norm_bias) {
    fprintf(stderr, "Failed to load self output weights\n");
    goto cleanup_bertoutput;
  }
  print_float_array(bertoutput.dense_weight, 5);
  print_float_array(bertoutput.dense_bias, 5);
  print_float_array(bertoutput.layer_norm_weight, 5);
  print_float_array(bertoutput.layer_norm_bias, 5);
  // ===============Encoder Atention=================
  BertAttention attention = {self_attention, self_output};

  int input_size;
  float *input_tensor = load_tensor("bins/layer0_attn_input.bin", &input_size);
  if (!input_tensor) {
    fprintf(stderr, "Failed to load input tensor\n");
    goto cleanup_self_output;
  }

  float *output = malloc(total_seq_hidden * sizeof(float));
  if (!output) {
    fprintf(stderr, "Memory allocation failed for output\n");
    free(input_tensor);
    goto cleanup_self_output;
  }

  attention_forward(attention, config, input_tensor, output);

  // ===============Encoder Intermediate=================
  float *inter_out = malloc(total_seq_hidden * sizeof(float));
  if (!inter_out) {
    fprintf(stderr, "Memory allocation failed for output\n");
    free(inter_out);
    goto cleanup_intermediate;
  }
  intermediate_forward(intermediate, output, inter_out);

  // ===============Encoder Output=================
  float *bout = malloc(total_seq_hidden * sizeof(float));
  if (!bout) {
    fprintf(stderr, "Memory allocation failed for output\n");
    free(bout);
    goto cleanup_bertoutput;
  }
  bert_output_forward(bertoutput, bout, input_tensor, output, 2, 128, 768);
  print_float_array(bout, 5);
  free(input_tensor);
  free(output);

cleanup_intermediate:
  free(intermediate.dense_bias);
  free(intermediate.dense_weight);
cleanup_self_output:
  free(self_output.layer_norm_bias);
  free(self_output.layer_norm_weight);
  free(self_output.dense_bias);
  free(self_output.dense_weight);

cleanup_bertoutput:
  free(bertoutput.layer_norm_bias);
  free(bertoutput.layer_norm_weight);
  free(bertoutput.dense_bias);
  free(bertoutput.dense_weight);

cleanup_self_attention:
  free(self_attention.value_bias);
  free(self_attention.key_bias);
  free(self_attention.query_bias);
  free(self_attention.value_weight);
  free(self_attention.key_weight);
  free(self_attention.query_weight);

  return 0;
}
