#include "attention/utils.c"
#include "encoder/bert_utils.c"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BATCH_SIZE 2
#define SEQ_LENGTH 128
#define HIDDEN_SIZE 768
#define NUM_HEADS 12
#define INTERMEDIATE_SIZE 3072
#define VOCAB_SIZE 30522
#define MAX_POSITION_EMBEDDINGS 512
#define TYPE_VOCAB_SIZE 2

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

typedef struct {
    float *dense_weight;      // [hidden_size, hidden_size]
    float *dense_bias;        // [hidden_size]
    float *layer_norm_weight; // [hidden_size]
    float *layer_norm_bias;   // [hidden_size]
    float dropout_prob;
    float layer_norm_eps;
} BertSelfOutput;

typedef struct {
    BertSelfAttention self;
    BertSelfOutput output;
} BertAttention;

typedef struct {
    float *dense_weight; // [hidden_size, intermediate_size]
    float *dense_bias;   // [intermediate_size]
} BertIntermediate;

typedef struct {
    float *dense_weight;      // [intermediate_size, hidden_size]
    float *dense_bias;        // [hidden_size]
    float *layer_norm_weight; // [hidden_size]
    float *layer_norm_bias;   // [hidden_size]
    float dropout_prob;
    float layer_norm_eps;
} BertOutput;

typedef struct {
    BertAttention attention;
    BertIntermediate intermediate;
    BertOutput output;
} BertLayer;

typedef struct {
    float *word_embeddings;        // [vocab_size, hidden_size]
    float *position_embeddings;    // [max_position_embeddings, hidden_size]
    float *token_type_embeddings;  // [type_vocab_size, hidden_size]
    float *layer_norm_weight;      // [hidden_size]
    float *layer_norm_bias;        // [hidden_size]
    float dropout_prob;
    float layer_norm_eps;
} BertEmbeddings;

typedef struct {
    BertLayer *layers;  // Array of layers
    int num_layers;
} BertEncoder;

typedef struct {
    float *dense_weight;  // [hidden_size, hidden_size]
    float *dense_bias;    // [hidden_size]
} BertPooler;

typedef struct {
    BertEmbeddings embeddings;
    BertEncoder encoder;
    BertPooler pooler;
} BertModel;

// Embedding layer functions
void load_embeddings(BertEmbeddings *embed, BertConfig config) {
    const char *paths[] = {
        "bins/weights/embeddings_word_embeddings_weight.bin",
        "bins/weights/embeddings_position_embeddings_weight.bin",
        "bins/weights/embeddings_token_type_embeddings_weight.bin",
        "bins/weights/embeddings_LayerNorm_weight.bin",
        "bins/weights/embeddings_LayerNorm_bias.bin"
    };
    int total_size;

    embed->word_embeddings = load_tensor(paths[0], &total_size);
    embed->position_embeddings = load_tensor(paths[1], &total_size);
    embed->token_type_embeddings = load_tensor(paths[2], &total_size);
    embed->layer_norm_weight = load_tensor(paths[3], &total_size);
    embed->layer_norm_bias = load_tensor(paths[4], &total_size);

    embed->dropout_prob = config.hidden_dropout_prob;
    embed->layer_norm_eps = config.layer_norm_eps;
}

void bert_embeddings_forward(BertEmbeddings embed, int *input_ids, float *output) {
    float *word_embed = malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));
    float *pos_embed = malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));
    float *token_embed = malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));
    float *combined = malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));

    // Word embeddings
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int t = 0; t < SEQ_LENGTH; t++) {
            int idx = input_ids[b * SEQ_LENGTH + t];
            memcpy(&word_embed[(b * SEQ_LENGTH + t) * HIDDEN_SIZE],
                   &embed.word_embeddings[idx * HIDDEN_SIZE],
                   HIDDEN_SIZE * sizeof(float));
        }
    }
    int total_size;
    float *expected = load_tensor("bins/word_embeddings_output.bin", &total_size);
    check_tensor(expected, word_embed, 5, "word embeddings");
    free(expected);

    // Position embeddings (simple range 0 to SEQ_LENGTH-1)
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int t = 0; t < SEQ_LENGTH; t++) {
            memcpy(&pos_embed[(b * SEQ_LENGTH + t) * HIDDEN_SIZE],
                   &embed.position_embeddings[t * HIDDEN_SIZE],
                   HIDDEN_SIZE * sizeof(float));
        }
    }
    expected = load_tensor("bins/position_embeddings_output.bin", &total_size);
    check_tensor(expected, pos_embed, 5, "position embeddings");
    free(expected);

    // Token type embeddings (assuming all zeros)
    for (int i = 0; i < BATCH_SIZE * SEQ_LENGTH; i++) {
        memcpy(&token_embed[i * HIDDEN_SIZE],
               embed.token_type_embeddings,
               HIDDEN_SIZE * sizeof(float));
    }
    expected = load_tensor("bins/token_type_embeddings_output.bin", &total_size);
    check_tensor(expected, token_embed, 5, "token type embeddings");
    free(expected);

    // Combine embeddings
    for (int i = 0; i < BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE; i++) {
        combined[i] = word_embed[i] + pos_embed[i] + token_embed[i];
    }
    expected = load_tensor("bins/combined_embeddings.bin", &total_size);
    check_tensor(expected, combined, 5, "combined embeddings");
    free(expected);

    // Layer normalization
    layernorm_forward_cpu(output, combined, embed.layer_norm_weight,
                         embed.layer_norm_bias, BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE,
                         embed.layer_norm_eps);

    expected = load_tensor("bins/ln_o.bin", &total_size);
    check_tensor(expected, output, 5, "embedding layer norm");
    free(expected);

    free(word_embed);
    free(pos_embed);
    free(token_embed);
    free(combined);
}

// Attention layer functions
void load_self_attention(BertSelfAttention *bsa, BertConfig config, int layer_idx) {
    char paths[6][128];
    sprintf(paths[0], "bins/weights/encoder_layer_%d_attention_self_query_weight.bin", layer_idx);
    sprintf(paths[1], "bins/weights/encoder_layer_%d_attention_self_query_bias.bin", layer_idx);
    sprintf(paths[2], "bins/weights/encoder_layer_%d_attention_self_key_weight.bin", layer_idx);
    sprintf(paths[3], "bins/weights/encoder_layer_%d_attention_self_key_bias.bin", layer_idx);
    sprintf(paths[4], "bins/weights/encoder_layer_%d_attention_self_value_weight.bin", layer_idx);
    sprintf(paths[5], "bins/weights/encoder_layer_%d_attention_self_value_bias.bin", layer_idx);

    int total_size;
    bsa->query_weight = load_tensor(paths[0], &total_size);
    bsa->query_bias = load_tensor(paths[1], &total_size);
    bsa->key_weight = load_tensor(paths[2], &total_size);
    bsa->key_bias = load_tensor(paths[3], &total_size);
    bsa->value_weight = load_tensor(paths[4], &total_size);
    bsa->value_bias = load_tensor(paths[5], &total_size);

    bsa->num_heads = config.num_attention_heads;
    bsa->head_dim = config.hidden_size / config.num_attention_heads;
    bsa->scale = 1.0f / sqrtf(bsa->head_dim);
}

void load_self_output(BertSelfOutput *bso, BertConfig config, int layer_idx) {
    char paths[4][128];
    sprintf(paths[0], "bins/weights/encoder_layer_%d_attention_output_dense_weight.bin", layer_idx);
    sprintf(paths[1], "bins/weights/encoder_layer_%d_attention_output_dense_bias.bin", layer_idx);
    sprintf(paths[2], "bins/weights/encoder_layer_%d_attention_output_LayerNorm_weight.bin", layer_idx);
    sprintf(paths[3], "bins/weights/encoder_layer_%d_attention_output_LayerNorm_bias.bin", layer_idx);

    int total_size;
    bso->dense_weight = load_tensor(paths[0], &total_size);
    bso->dense_bias = load_tensor(paths[1], &total_size);
    bso->layer_norm_weight = load_tensor(paths[2], &total_size);
    bso->layer_norm_bias = load_tensor(paths[3], &total_size);

    bso->dropout_prob = config.hidden_dropout_prob;
    bso->layer_norm_eps = config.layer_norm_eps;
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float *out, float *inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

void bert_attention_forward(BertSelfAttention bsa, float *input, float *output, int layer_idx) {
    const int total_seq_hidden = BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE;
    float *q_proj = malloc(total_seq_hidden * sizeof(float));
    float *k_proj = malloc(total_seq_hidden * sizeof(float));
    float *v_proj = malloc(total_seq_hidden * sizeof(float));

    linear_projection(input, bsa.query_weight, bsa.query_bias, q_proj,
                     BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE, HIDDEN_SIZE);
    linear_projection(input, bsa.key_weight, bsa.key_bias, k_proj,
                     BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE, HIDDEN_SIZE);
    linear_projection(input, bsa.value_weight, bsa.value_bias, v_proj,
                     BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE, HIDDEN_SIZE);

    const int total_head_seq_dim = BATCH_SIZE * bsa.num_heads * SEQ_LENGTH * bsa.head_dim;
    float *q_reshaped = malloc(total_head_seq_dim * sizeof(float));
    float *k_reshaped = malloc(total_head_seq_dim * sizeof(float));
    float *v_reshaped = malloc(total_head_seq_dim * sizeof(float));

    reshape_for_multihead(q_proj, q_reshaped, BATCH_SIZE, SEQ_LENGTH, bsa.num_heads, bsa.head_dim);
    reshape_for_multihead(k_proj, k_reshaped, BATCH_SIZE, SEQ_LENGTH, bsa.num_heads, bsa.head_dim);
    reshape_for_multihead(v_proj, v_reshaped, BATCH_SIZE, SEQ_LENGTH, bsa.num_heads, bsa.head_dim);

    const int attn_size = BATCH_SIZE * bsa.num_heads * SEQ_LENGTH * SEQ_LENGTH;
    float *attn_scores = malloc(attn_size * sizeof(float));
    float *attn_scaled = malloc(attn_size * sizeof(float));
    float *attn_softmax = malloc(attn_size * sizeof(float));

    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int h = 0; h < bsa.num_heads; h++) {
            float *q_ptr = q_reshaped + (b * bsa.num_heads + h) * SEQ_LENGTH * bsa.head_dim;
            float *k_ptr = k_reshaped + (b * bsa.num_heads + h) * SEQ_LENGTH * bsa.head_dim;
            float *v_ptr = v_reshaped + (b * bsa.num_heads + h) * SEQ_LENGTH * bsa.head_dim;
            float *scores_ptr = attn_scores + (b * bsa.num_heads + h) * SEQ_LENGTH * SEQ_LENGTH;
            float *scaled_ptr = attn_scaled + (b * bsa.num_heads + h) * SEQ_LENGTH * SEQ_LENGTH;
            float *softmax_ptr = attn_softmax + (b * bsa.num_heads + h) * SEQ_LENGTH * SEQ_LENGTH;

            matmul_transpose(q_ptr, k_ptr, scores_ptr, SEQ_LENGTH, bsa.head_dim, SEQ_LENGTH);
            for (int i = 0; i < SEQ_LENGTH * SEQ_LENGTH; i++) {
                scaled_ptr[i] = scores_ptr[i] * bsa.scale;
            }
            softmax(scaled_ptr, softmax_ptr, SEQ_LENGTH, SEQ_LENGTH);
            matmul(softmax_ptr, v_ptr, output + (b * bsa.num_heads + h) * SEQ_LENGTH * bsa.head_dim,
                   SEQ_LENGTH, SEQ_LENGTH, bsa.head_dim);
        }
    }

    float *context = malloc(total_seq_hidden * sizeof(float));
    reshape_from_multihead(output, context, BATCH_SIZE, SEQ_LENGTH, bsa.num_heads, bsa.head_dim);
    memcpy(output, context, total_seq_hidden * sizeof(float));

    char path[128];
    sprintf(path, "bins/layer%d_context.bin", layer_idx);
    int total_size;
    float *expected = load_tensor(path, &total_size);
    check_tensor(expected, output, 5, "attention context");
    free(expected);

    free(context);
    free(attn_softmax);
    free(attn_scaled);
    free(attn_scores);
    free(v_reshaped);
    free(k_reshaped);
    free(q_reshaped);
    free(v_proj);
    free(k_proj);
    free(q_proj);
}

void bert_selfoutput_forward(BertSelfOutput bso, float *output, float *hidden_states,
                            float *input_tensor, int layer_idx) {
    float *temp_buffer = malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));
    matmul_forward(temp_buffer, hidden_states, bso.dense_weight, bso.dense_bias,
                  BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE, HIDDEN_SIZE);
    apply_dropout(temp_buffer, temp_buffer, bso.dropout_prob, BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE);

    float *residual = malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));
    add_tensors(residual, temp_buffer, input_tensor, BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE);
    layernorm_forward_cpu(output, residual, bso.layer_norm_weight, bso.layer_norm_bias,
                        BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE, bso.layer_norm_eps);

    char path[128];
    sprintf(path, "bins/layer%d_self_output_layernorm.bin", layer_idx);
    int total_size;
    float *expected = load_tensor(path, &total_size);
    check_tensor(expected, output, 5, "self output");
    free(expected);

    free(residual);
    free(temp_buffer);
}

// Intermediate and Output layers
void load_intermediate(BertIntermediate *intermediate, BertConfig config, int layer_idx) {
    char paths[2][128];
    sprintf(paths[0], "bins/weights/encoder_layer_%d_intermediate_dense_weight.bin", layer_idx);
    sprintf(paths[1], "bins/weights/encoder_layer_%d_intermediate_dense_bias.bin", layer_idx);

    int total_size;
    intermediate->dense_weight = load_tensor(paths[0], &total_size);
    intermediate->dense_bias = load_tensor(paths[1], &total_size);
}

void load_output(BertOutput *bo, BertConfig config, int layer_idx) {
    char paths[4][128];
    sprintf(paths[0], "bins/weights/encoder_layer_%d_output_dense_weight.bin", layer_idx);
    sprintf(paths[1], "bins/weights/encoder_layer_%d_output_dense_bias.bin", layer_idx);
    sprintf(paths[2], "bins/weights/encoder_layer_%d_output_LayerNorm_weight.bin", layer_idx);
    sprintf(paths[3], "bins/weights/encoder_layer_%d_output_LayerNorm_bias.bin", layer_idx);

    int total_size;
    bo->dense_weight = load_tensor(paths[0], &total_size);
    bo->dense_bias = load_tensor(paths[1], &total_size);
    bo->layer_norm_weight = load_tensor(paths[2], &total_size);
    bo->layer_norm_bias = load_tensor(paths[3], &total_size);

    bo->dropout_prob = config.hidden_dropout_prob;
    bo->layer_norm_eps = config.layer_norm_eps;
}

void intermediate_forward(BertIntermediate intermediate, float *input, float *output, int layer_idx) {
    matmul_forward(output, input, intermediate.dense_weight, intermediate.dense_bias,
                  BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    gelu_forward(output, output, BATCH_SIZE * SEQ_LENGTH * INTERMEDIATE_SIZE);

    char path[128];
    sprintf(path, "bins/layer%d_intermediate_activation.bin", layer_idx);
    int total_size;
    float *expected = load_tensor(path, &total_size);
    check_tensor(expected, output, 5, "intermediate output");
    free(expected);
}

void bert_output_forward(BertOutput bo, float *output, float *hidden_states,
                        float *input_tensor, int layer_idx) {
    float *temp_buffer = malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));
    matmul_forward(temp_buffer, hidden_states, bo.dense_weight, bo.dense_bias,
                  BATCH_SIZE, SEQ_LENGTH, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    apply_dropout(temp_buffer, temp_buffer, bo.dropout_prob, BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE);

    float *residual = malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));
    add_tensors(residual, temp_buffer, input_tensor, BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE);
    layernorm_forward_cpu(output, residual, bo.layer_norm_weight, bo.layer_norm_bias,
                         BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE, bo.layer_norm_eps);

    char path[128];
    sprintf(path, "bins/layer%d_output_layernorm.bin", layer_idx);
    int total_size;
    float *expected = load_tensor(path, &total_size);
    check_tensor(expected, output, 5, "output layer norm");
    free(expected);

    free(residual);
    free(temp_buffer);
}

// Layer and Encoder functions
void bert_layer_forward(BertLayer layer, float *input, float *output, int layer_idx) {
    float *attn_output = malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));
    float *intermediate_output = malloc(BATCH_SIZE * SEQ_LENGTH * INTERMEDIATE_SIZE * sizeof(float));

    bert_attention_forward(layer.attention.self, input, attn_output, layer_idx);
    bert_selfoutput_forward(layer.attention.output, attn_output, attn_output, input, layer_idx);
    intermediate_forward(layer.intermediate, attn_output, intermediate_output, layer_idx);
    bert_output_forward(layer.output, output, intermediate_output, attn_output, layer_idx);

    free(intermediate_output);
    free(attn_output);
}

void bert_encoder_forward(BertEncoder encoder, float *input, float *output) {
    float *current = malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));
    memcpy(current, input, BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));

    for (int i = 0; i < encoder.num_layers; i++) {
        bert_layer_forward(encoder.layers[i], current, output, i);
        memcpy(current, output, BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));
    }

    memcpy(output, current, BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));
    free(current);

    int total_size;
    float *expected = load_tensor("bins/encoder_output.bin", &total_size);
    check_tensor(expected, output, 5, "encoder output");
    free(expected);
}

// Pooler functions
void load_pooler(BertPooler *pooler, BertConfig config) {
    const char *paths[] = {
        "bins/weights/pooler_dense_weight.bin",
        "bins/weights/pooler_dense_bias.bin"
    };
    int total_size;
    pooler->dense_weight = load_tensor(paths[0], &total_size);
    pooler->dense_bias = load_tensor(paths[1], &total_size);
}

void bert_pooler_forward(BertPooler pooler, float *input, float *output) {
    float *first_token = malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    for (int b = 0; b < BATCH_SIZE; b++) {
        memcpy(&first_token[b * HIDDEN_SIZE],
               &input[b * SEQ_LENGTH * HIDDEN_SIZE],
               HIDDEN_SIZE * sizeof(float));
    }

    matmul_forward(output, first_token, pooler.dense_weight, pooler.dense_bias,
                  BATCH_SIZE, 1, HIDDEN_SIZE, HIDDEN_SIZE);

    // Apply tanh
    for (int i = 0; i < BATCH_SIZE * HIDDEN_SIZE; i++) {
        output[i] = tanhf(output[i]);
    }

    int total_size;
    float *expected = load_tensor("bins/model_pooled_output.bin", &total_size);
    check_tensor(expected, output, 5, "pooler output");
    free(expected);
    free(first_token);
}

// Main model functions
void bert_model_forward(BertModel model, int *input_ids, float *sequence_output, float *pooled_output) {
    float *embedding_output = malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));
    bert_embeddings_forward(model.embeddings, input_ids, embedding_output);
    bert_encoder_forward(model.encoder, embedding_output, sequence_output);
    bert_pooler_forward(model.pooler, sequence_output, pooled_output);
    free(embedding_output);
}

int main() {
    BertConfig config = default_config();
    BertModel model;

    // Initialize embeddings
    load_embeddings(&model.embeddings, config);

    // Initialize encoder
    model.encoder.num_layers = config.num_hidden_layers;
    model.encoder.layers = malloc(model.encoder.num_layers * sizeof(BertLayer));
    for (int i = 0; i < model.encoder.num_layers; i++) {
        load_self_attention(&model.encoder.layers[i].attention.self, config, i);
        load_self_output(&model.encoder.layers[i].attention.output, config, i);
        load_intermediate(&model.encoder.layers[i].intermediate, config, i);
        load_output(&model.encoder.layers[i].output, config, i);
    }

    // Initialize pooler
    load_pooler(&model.pooler, config);

    // Load input
    int total_size;
    int *input_ids = (int *)load_tensor("bins/input_ids.bin", &total_size);

    // Allocate outputs
    float *sequence_output = malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));
    float *pooled_output = malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));

    // Run model
    bert_model_forward(model, input_ids, sequence_output, pooled_output);

    // Verify final outputs
    float *expected_seq = load_tensor("bins/model_sequence_output.bin", &total_size);
    check_tensor(expected_seq, sequence_output, 5, "final sequence output");
    free(expected_seq);

    // Cleanup
    free(sequence_output);
    free(pooled_output);
    free(input_ids);

    free(model.pooler.dense_bias);
    free(model.pooler.dense_weight);

    for (int i = 0; i < model.encoder.num_layers; i++) {
        free(model.encoder.layers[i].output.layer_norm_bias);
        free(model.encoder.layers[i].output.layer_norm_weight);
        free(model.encoder.layers[i].output.dense_bias);
        free(model.encoder.layers[i].output.dense_weight);
        free(model.encoder.layers[i].intermediate.dense_bias);
        free(model.encoder.layers[i].intermediate.dense_weight);
        free(model.encoder.layers[i].attention.output.layer_norm_bias);
        free(model.encoder.layers[i].attention.output.layer_norm_weight);
        free(model.encoder.layers[i].attention.output.dense_bias);
        free(model.encoder.layers[i].attention.output.dense_weight);
        free(model.encoder.layers[i].attention.self.value_bias);
        free(model.encoder.layers[i].attention.self.key_bias);
        free(model.encoder.layers[i].attention.self.query_bias);
        free(model.encoder.layers[i].attention.self.value_weight);
        free(model.encoder.layers[i].attention.self.key_weight);
        free(model.encoder.layers[i].attention.self.query_weight);
    }
    free(model.encoder.layers);

    free(model.embeddings.layer_norm_bias);
    free(model.embeddings.layer_norm_weight);
    free(model.embeddings.token_type_embeddings);
    free(model.embeddings.position_embeddings);
    free(model.embeddings.word_embeddings);

    return 0;
}
