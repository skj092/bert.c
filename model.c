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
#define EXPECTED_HIDDEN_SIZE 768
#define EXPECTED_NUM_LAYERS 12
#define EXPECTED_NUM_HEADS 12
#define EXPECTED_INTERMEDIATE_SIZE 3072
#define EXPECTED_VOCAB_SIZE 30522
#define EXPECTED_MAX_POS_EMBEDDINGS 512
#define EXPECTED_TYPE_VOCAB_SIZE 2

typedef struct {
    float *word_embeddings;
    float *position_embeddings;
    float *token_type_embeddings;
    float *layer_norm_weight;
    float *layer_norm_bias;
    float dropout_prob;
    float layer_norm_eps;
} BertEmbeddings;

typedef struct {
    float *query_weight;
    float *query_bias;
    float *key_weight;
    float *key_bias;
    float *value_weight;
    float *value_bias;
    int num_heads;
    int head_dim;
    float scale;
} BertSelfAttention;

typedef struct {
    float *dense_weight;
    float *dense_bias;
    float *layer_norm_weight;
    float *layer_norm_bias;
    float dropout_prob;
    float layer_norm_eps;
} BertSelfOutput;

typedef struct {
    BertSelfAttention self;
    BertSelfOutput output;
} BertAttention;

typedef struct {
    float *dense_weight;
    float *dense_bias;
} BertIntermediate;

typedef struct {
    float *dense_weight;
    float *dense_bias;
    float *layer_norm_weight;
    float *layer_norm_bias;
    float dropout_prob;
    float layer_norm_eps;
} BertOutput;

typedef struct {
    BertAttention attention;
    BertIntermediate intermediate;
    BertOutput output;
} BertLayer;

typedef struct {
    BertLayer *layers;
    int num_layers;
} BertEncoder;

typedef struct {
    float *dense_weight;
    float *dense_bias;
} BertPooler;

typedef struct {
    BertEmbeddings embeddings;
    BertEncoder encoder;
    BertPooler pooler;
    int hidden_size;
    int num_attention_heads;
    int intermediate_size;
    int vocab_size;
    int max_position_embeddings;
    int type_vocab_size;
} BertModel;

void bert_build_from_checkpoint(BertModel *model, const char *checkpoint_path) {
    FILE *f = fopen(checkpoint_path, "rb");
    if (!f) {
        fprintf(stderr, "Error opening checkpoint file %s\n", checkpoint_path);
        exit(1);
    }
    printf("Opened checkpoint file\n");

    char version[6];
    fread(version, 1, 5, f);
    version[5] = '\0';
    if (strcmp(version, "BERTv") != 0) {
        fprintf(stderr, "Invalid checkpoint version: %s\n", version);
        fclose(f);
        exit(1);
    }
    printf("Checkpoint version verified\n");

    int config[7];
    size_t read_count = fread(config, sizeof(int), 7, f);
    if (read_count != 7) {
        fprintf(stderr, "Failed to read config, read %zu elements\n", read_count);
        fclose(f);
        exit(1);
    }
    model->hidden_size = config[0];
    model->encoder.num_layers = config[1];
    model->num_attention_heads = config[2];
    model->intermediate_size = config[3];
    model->vocab_size = config[4];
    model->max_position_embeddings = config[5];
    model->type_vocab_size = config[6];
    printf("Config loaded: hidden_size=%d, num_layers=%d, num_heads=%d, intermediate_size=%d, vocab_size=%d, max_pos=%d, type_vocab=%d\n",
           model->hidden_size, model->encoder.num_layers, model->num_attention_heads,
           model->intermediate_size, model->vocab_size, model->max_position_embeddings, model->type_vocab_size);

    // Validate config
    if (model->hidden_size != EXPECTED_HIDDEN_SIZE ||
        model->encoder.num_layers != EXPECTED_NUM_LAYERS ||
        model->num_attention_heads != EXPECTED_NUM_HEADS ||
        model->intermediate_size != EXPECTED_INTERMEDIATE_SIZE ||
        model->vocab_size != EXPECTED_VOCAB_SIZE ||
        model->max_position_embeddings != EXPECTED_MAX_POS_EMBEDDINGS ||
        model->type_vocab_size != EXPECTED_TYPE_VOCAB_SIZE) {
        fprintf(stderr, "Invalid config values detected!\n");
        fclose(f);
        exit(1);
    }

    model->encoder.layers = malloc(model->encoder.num_layers * sizeof(BertLayer));
    if (!model->encoder.layers) {
        fprintf(stderr, "Failed to allocate memory for %d layers\n", model->encoder.num_layers);
        fclose(f);
        exit(1);
    }
    printf("Allocated %d layers\n", model->encoder.num_layers);

    while (!feof(f)) {
        int name_len;
        if (fread(&name_len, sizeof(int), 1, f) != 1) break;

        char *name = malloc(name_len + 1);
        if (!name) {
            fprintf(stderr, "Failed to allocate memory for parameter name\n");
            fclose(f);
            exit(1);
        }
        fread(name, 1, name_len, f);
        name[name_len] = '\0';

        int shape_len;
        fread(&shape_len, sizeof(int), 1, f);
        int *shape = malloc(shape_len * sizeof(int));
        if (!shape) {
            fprintf(stderr, "Failed to allocate memory for shape\n");
            free(name);
            fclose(f);
            exit(1);
        }
        fread(shape, sizeof(int), shape_len, f);

        int total_size = 1;
        for (int i = 0; i < shape_len; i++) total_size *= shape[i];
        float *data = malloc(total_size * sizeof(float));
        if (!data) {
            fprintf(stderr, "Failed to allocate memory for %s (%d floats)\n", name, total_size);
            free(name);
            free(shape);
            fclose(f);
            exit(1);
        }
        fread(data, sizeof(float), total_size, f);
        printf("Loaded parameter: %s, size=%d\n", name, total_size);

        if (strcmp(name, "embeddings.word_embeddings.weight") == 0) {
            model->embeddings.word_embeddings = data;
        } else if (strcmp(name, "embeddings.position_embeddings.weight") == 0) {
            model->embeddings.position_embeddings = data;
        } else if (strcmp(name, "embeddings.token_type_embeddings.weight") == 0) {
            model->embeddings.token_type_embeddings = data;
        } else if (strcmp(name, "embeddings.LayerNorm.weight") == 0) {
            model->embeddings.layer_norm_weight = data;
        } else if (strcmp(name, "embeddings.LayerNorm.bias") == 0) {
            model->embeddings.layer_norm_bias = data;
        } else if (strcmp(name, "pooler.dense.weight") == 0) {
            model->pooler.dense_weight = data;
        } else if (strcmp(name, "pooler.dense.bias") == 0) {
            model->pooler.dense_bias = data;
        } else {
            int layer_idx;
            if (sscanf(name, "encoder.layer.%d", &layer_idx) == 1 && layer_idx < model->encoder.num_layers) {
                BertLayer *layer = &model->encoder.layers[layer_idx];
                if (strstr(name, "attention.self.query.weight")) {
                    layer->attention.self.query_weight = data;
                } else if (strstr(name, "attention.self.query.bias")) {
                    layer->attention.self.query_bias = data;
                } else if (strstr(name, "attention.self.key.weight")) {
                    layer->attention.self.key_weight = data;
                } else if (strstr(name, "attention.self.key.bias")) {
                    layer->attention.self.key_bias = data;
                } else if (strstr(name, "attention.self.value.weight")) {
                    layer->attention.self.value_weight = data;
                } else if (strstr(name, "attention.self.value.bias")) {
                    layer->attention.self.value_bias = data;
                } else if (strstr(name, "attention.output.dense.weight")) {
                    layer->attention.output.dense_weight = data;
                } else if (strstr(name, "attention.output.dense.bias")) {
                    layer->attention.output.dense_bias = data;
                } else if (strstr(name, "attention.output.LayerNorm.weight")) {
                    layer->attention.output.layer_norm_weight = data;
                } else if (strstr(name, "attention.output.LayerNorm.bias")) {
                    layer->attention.output.layer_norm_bias = data;
                } else if (strstr(name, "intermediate.dense.weight")) {
                    layer->intermediate.dense_weight = data;
                } else if (strstr(name, "intermediate.dense.bias")) {
                    layer->intermediate.dense_bias = data;
                } else if (strstr(name, "output.dense.weight")) {
                    layer->output.dense_weight = data;
                } else if (strstr(name, "output.dense.bias")) {
                    layer->output.dense_bias = data;
                } else if (strstr(name, "output.LayerNorm.weight")) {
                    layer->output.layer_norm_weight = data;
                } else if (strstr(name, "output.LayerNorm.bias")) {
                    layer->output.layer_norm_bias = data;
                }
                layer->attention.self.num_heads = model->num_attention_heads;
                layer->attention.self.head_dim = model->hidden_size / model->num_attention_heads;
                layer->attention.self.scale = 1.0f / sqrtf(layer->attention.self.head_dim);
                layer->attention.output.dropout_prob = 0.1f;
                layer->attention.output.layer_norm_eps = 1e-12f;
                layer->output.dropout_prob = 0.1f;
                layer->output.layer_norm_eps = 1e-12f;
            } else {
                free(data);
            }
        }
        free(name);
        free(shape);
    }
    model->embeddings.dropout_prob = 0.1f;
    model->embeddings.layer_norm_eps = 1e-12f;
    fclose(f);
    printf("Checkpoint loading completed\n");
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float *out, float *inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

void bert_embeddings_forward(BertEmbeddings embed, int *input_ids, float *output, int hidden_size) {
    if (!input_ids || !embed.word_embeddings) {
        fprintf(stderr, "Null pointer in embeddings forward\n");
        exit(1);
    }
    printf("Starting embeddings forward\n");

    float *word_embed = malloc(BATCH_SIZE * SEQ_LENGTH * hidden_size * sizeof(float));
    float *pos_embed = malloc(BATCH_SIZE * SEQ_LENGTH * hidden_size * sizeof(float));
    float *token_embed = malloc(BATCH_SIZE * SEQ_LENGTH * hidden_size * sizeof(float));
    float *combined = malloc(BATCH_SIZE * SEQ_LENGTH * hidden_size * sizeof(float));

    if (!word_embed || !pos_embed || !token_embed || !combined) {
        fprintf(stderr, "Memory allocation failed in embeddings\n");
        free(word_embed); free(pos_embed); free(token_embed); free(combined);
        exit(1);
    }

    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int t = 0; t < SEQ_LENGTH; t++) {
            int idx = input_ids[b * SEQ_LENGTH + t];
            if (idx >= EXPECTED_VOCAB_SIZE) {
                fprintf(stderr, "Invalid input_id: %d\n", idx);
                exit(1);
            }
            memcpy(&word_embed[(b * SEQ_LENGTH + t) * hidden_size],
                   &embed.word_embeddings[idx * hidden_size],
                   hidden_size * sizeof(float));
        }
    }
    printf("Word embeddings computed\n");
    int total_size;
    float *expected = load_tensor("bins/word_embeddings_output.bin", &total_size);
    if (expected) {
        check_tensor(expected, word_embed, 5, "word embeddings");
        free(expected);
    }

    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int t = 0; t < SEQ_LENGTH; t++) {
            memcpy(&pos_embed[(b * SEQ_LENGTH + t) * hidden_size],
                   &embed.position_embeddings[t * hidden_size],
                   hidden_size * sizeof(float));
        }
    }
    printf("Position embeddings computed\n");
    expected = load_tensor("bins/position_embeddings_output.bin", &total_size);
    if (expected) {
        check_tensor(expected, pos_embed, 5, "position embeddings");
        free(expected);
    }

    for (int i = 0; i < BATCH_SIZE * SEQ_LENGTH; i++) {
        memcpy(&token_embed[i * hidden_size],
               embed.token_type_embeddings,
               hidden_size * sizeof(float));
    }
    printf("Token type embeddings computed\n");
    expected = load_tensor("bins/token_type_embeddings_output.bin", &total_size);
    if (expected) {
        check_tensor(expected, token_embed, 5, "token type embeddings");
        free(expected);
    }

    for (int i = 0; i < BATCH_SIZE * SEQ_LENGTH * hidden_size; i++) {
        combined[i] = word_embed[i] + pos_embed[i] + token_embed[i];
    }
    printf("Embeddings combined\n");
    expected = load_tensor("bins/combined_embeddings.bin", &total_size);
    if (expected) {
        check_tensor(expected, combined, 5, "combined embeddings");
        free(expected);
    }

    layernorm_forward_cpu(output, combined, embed.layer_norm_weight,
                         embed.layer_norm_bias, BATCH_SIZE, SEQ_LENGTH, hidden_size,
                         embed.layer_norm_eps);
    printf("Layer norm completed\n");
    expected = load_tensor("bins/ln_o.bin", &total_size);
    if (expected) {
        check_tensor(expected, output, 5, "embedding layer norm");
        free(expected);
    }

    free(word_embed);
    free(pos_embed);
    free(token_embed);
    free(combined);
}

void bert_attention_forward(BertSelfAttention bsa, float *input, float *output, int layer_idx, int hidden_size) {
    printf("Starting attention forward for layer %d\n", layer_idx);
    const int total_seq_hidden = BATCH_SIZE * SEQ_LENGTH * hidden_size;
    float *q_proj = malloc(total_seq_hidden * sizeof(float));
    float *k_proj = malloc(total_seq_hidden * sizeof(float));
    float *v_proj = malloc(total_seq_hidden * sizeof(float));
    if (!q_proj || !k_proj || !v_proj) {
        fprintf(stderr, "Memory allocation failed in attention\n");
        exit(1);
    }

    linear_projection(input, bsa.query_weight, bsa.query_bias, q_proj,
                     BATCH_SIZE, SEQ_LENGTH, hidden_size, hidden_size);
    linear_projection(input, bsa.key_weight, bsa.key_bias, k_proj,
                     BATCH_SIZE, SEQ_LENGTH, hidden_size, hidden_size);
    linear_projection(input, bsa.value_weight, bsa.value_bias, v_proj,
                     BATCH_SIZE, SEQ_LENGTH, hidden_size, hidden_size);

    const int total_head_seq_dim = BATCH_SIZE * bsa.num_heads * SEQ_LENGTH * bsa.head_dim;
    float *q_reshaped = malloc(total_head_seq_dim * sizeof(float));
    float *k_reshaped = malloc(total_head_seq_dim * sizeof(float));
    float *v_reshaped = malloc(total_head_seq_dim * sizeof(float));
    if (!q_reshaped || !k_reshaped || !v_reshaped) {
        fprintf(stderr, "Memory allocation failed in attention reshape\n");
        exit(1);
    }

    reshape_for_multihead(q_proj, q_reshaped, BATCH_SIZE, SEQ_LENGTH, bsa.num_heads, bsa.head_dim);
    reshape_for_multihead(k_proj, k_reshaped, BATCH_SIZE, SEQ_LENGTH, bsa.num_heads, bsa.head_dim);
    reshape_for_multihead(v_proj, v_reshaped, BATCH_SIZE, SEQ_LENGTH, bsa.num_heads, bsa.head_dim);

    const int attn_size = BATCH_SIZE * bsa.num_heads * SEQ_LENGTH * SEQ_LENGTH;
    float *attn_scores = malloc(attn_size * sizeof(float));
    float *attn_scaled = malloc(attn_size * sizeof(float));
    float *attn_softmax = malloc(attn_size * sizeof(float));
    if (!attn_scores || !attn_scaled || !attn_softmax) {
        fprintf(stderr, "Memory allocation failed in attention scores\n");
        exit(1);
    }

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
    if (!context) {
        fprintf(stderr, "Memory allocation failed for context\n");
        exit(1);
    }
    reshape_from_multihead(output, context, BATCH_SIZE, SEQ_LENGTH, bsa.num_heads, bsa.head_dim);
    memcpy(output, context, total_seq_hidden * sizeof(float));

    char path[128];
    sprintf(path, "bins/layer%d_context.bin", layer_idx);
    int total_size;
    float *expected = load_tensor(path, &total_size);
    if (expected) {
        check_tensor(expected, output, 5, "attention context");
        free(expected);
    }

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
    printf("Attention forward completed for layer %d\n", layer_idx);
}

void bert_selfoutput_forward(BertSelfOutput bso, float *output, float *hidden_states,
                            float *input_tensor, int layer_idx, int hidden_size) {
    printf("Starting self output forward for layer %d\n", layer_idx);
    float *temp_buffer = malloc(BATCH_SIZE * SEQ_LENGTH * hidden_size * sizeof(float));
    if (!temp_buffer) {
        fprintf(stderr, "Memory allocation failed in self output\n");
        exit(1);
    }
    matmul_forward(temp_buffer, hidden_states, bso.dense_weight, bso.dense_bias,
                  BATCH_SIZE, SEQ_LENGTH, hidden_size, hidden_size);
    apply_dropout(temp_buffer, temp_buffer, bso.dropout_prob, BATCH_SIZE * SEQ_LENGTH * hidden_size);

    float *residual = malloc(BATCH_SIZE * SEQ_LENGTH * hidden_size * sizeof(float));
    if (!residual) {
        fprintf(stderr, "Memory allocation failed for residual\n");
        exit(1);
    }
    add_tensors(residual, temp_buffer, input_tensor, BATCH_SIZE * SEQ_LENGTH * hidden_size);
    layernorm_forward_cpu(output, residual, bso.layer_norm_weight, bso.layer_norm_bias,
                        BATCH_SIZE, SEQ_LENGTH, hidden_size, bso.layer_norm_eps);

    char path[128];
    sprintf(path, "bins/layer%d_self_output_layernorm.bin", layer_idx);
    int total_size;
    float *expected = load_tensor(path, &total_size);
    if (expected) {
        check_tensor(expected, output, 5, "self output");
        free(expected);
    }

    free(residual);
    free(temp_buffer);
    printf("Self output forward completed for layer %d\n", layer_idx);
}

void intermediate_forward(BertIntermediate intermediate, float *input, float *output, int layer_idx, int hidden_size, int intermediate_size) {
    printf("Starting intermediate forward for layer %d\n", layer_idx);
    matmul_forward(output, input, intermediate.dense_weight, intermediate.dense_bias,
                  BATCH_SIZE, SEQ_LENGTH, hidden_size, intermediate_size);
    gelu_forward(output, output, BATCH_SIZE * SEQ_LENGTH * intermediate_size);

    char path[128];
    sprintf(path, "bins/layer%d_intermediate_activation.bin", layer_idx);
    int total_size;
    float *expected = load_tensor(path, &total_size);
    if (expected) {
        check_tensor(expected, output, 5, "intermediate output");
        free(expected);
    }
    printf("Intermediate forward completed for layer %d\n", layer_idx);
}

void bert_output_forward(BertOutput bo, float *output, float *hidden_states,
                        float *input_tensor, int layer_idx, int hidden_size, int intermediate_size) {
    printf("Starting output forward for layer %d\n", layer_idx);
    float *temp_buffer = malloc(BATCH_SIZE * SEQ_LENGTH * hidden_size * sizeof(float));
    if (!temp_buffer) {
        fprintf(stderr, "Memory allocation failed in output\n");
        exit(1);
    }
    matmul_forward(temp_buffer, hidden_states, bo.dense_weight, bo.dense_bias,
                  BATCH_SIZE, SEQ_LENGTH, intermediate_size, hidden_size);
    apply_dropout(temp_buffer, temp_buffer, bo.dropout_prob, BATCH_SIZE * SEQ_LENGTH * hidden_size);

    float *residual = malloc(BATCH_SIZE * SEQ_LENGTH * hidden_size * sizeof(float));
    if (!residual) {
        fprintf(stderr, "Memory allocation failed for residual\n");
        exit(1);
    }
    add_tensors(residual, temp_buffer, input_tensor, BATCH_SIZE * SEQ_LENGTH * hidden_size);
    layernorm_forward_cpu(output, residual, bo.layer_norm_weight, bo.layer_norm_bias,
                         BATCH_SIZE, SEQ_LENGTH, hidden_size, bo.layer_norm_eps);

    char path[128];
    sprintf(path, "bins/layer%d_output_layernorm.bin", layer_idx);
    int total_size;
    float *expected = load_tensor(path, &total_size);
    if (expected) {
        check_tensor(expected, output, 5, "output layer norm");
        free(expected);
    }

    free(residual);
    free(temp_buffer);
    printf("Output forward completed for layer %d\n", layer_idx);
}

void bert_layer_forward(BertLayer layer, float *input, float *output, int layer_idx, int hidden_size, int intermediate_size) {
    printf("Starting layer forward for layer %d\n", layer_idx);
    float *attn_output = malloc(BATCH_SIZE * SEQ_LENGTH * hidden_size * sizeof(float));
    float *intermediate_output = malloc(BATCH_SIZE * SEQ_LENGTH * intermediate_size * sizeof(float));
    if (!attn_output || !intermediate_output) {
        fprintf(stderr, "Memory allocation failed in layer forward\n");
        exit(1);
    }

    bert_attention_forward(layer.attention.self, input, attn_output, layer_idx, hidden_size);
    bert_selfoutput_forward(layer.attention.output, attn_output, attn_output, input, layer_idx, hidden_size);
    intermediate_forward(layer.intermediate, attn_output, intermediate_output, layer_idx, hidden_size, intermediate_size);
    bert_output_forward(layer.output, output, intermediate_output, attn_output, layer_idx, hidden_size, intermediate_size);

    free(intermediate_output);
    free(attn_output);
    printf("Layer forward completed for layer %d\n", layer_idx);
}

void bert_encoder_forward(BertEncoder encoder, float *input, float *output, int hidden_size, int intermediate_size) {
    printf("Starting encoder forward\n");
    float *current = malloc(BATCH_SIZE * SEQ_LENGTH * hidden_size * sizeof(float));
    if (!current) {
        fprintf(stderr, "Memory allocation failed in encoder\n");
        exit(1);
    }
    memcpy(current, input, BATCH_SIZE * SEQ_LENGTH * hidden_size * sizeof(float));

    for (int i = 0; i < encoder.num_layers; i++) {
        bert_layer_forward(encoder.layers[i], current, output, i, hidden_size, intermediate_size);
        memcpy(current, output, BATCH_SIZE * SEQ_LENGTH * hidden_size * sizeof(float));
    }

    memcpy(output, current, BATCH_SIZE * SEQ_LENGTH * hidden_size * sizeof(float));
    free(current);

    int total_size;
    float *expected = load_tensor("bins/encoder_output.bin", &total_size);
    if (expected) {
        check_tensor(expected, output, 5, "encoder output");
        free(expected);
    }
    printf("Encoder forward completed\n");
}

void bert_pooler_forward(BertPooler pooler, float *input, float *output, int hidden_size) {
    printf("Starting pooler forward\n");
    float *first_token = malloc(BATCH_SIZE * hidden_size * sizeof(float));
    if (!first_token) {
        fprintf(stderr, "Memory allocation failed in pooler\n");
        exit(1);
    }
    for (int b = 0; b < BATCH_SIZE; b++) {
        memcpy(&first_token[b * hidden_size],
               &input[b * SEQ_LENGTH * hidden_size],
               hidden_size * sizeof(float));
    }

    matmul_forward(output, first_token, pooler.dense_weight, pooler.dense_bias,
                  BATCH_SIZE, 1, hidden_size, hidden_size);

    for (int i = 0; i < BATCH_SIZE * hidden_size; i++) {
        output[i] = tanhf(output[i]);
    }

    int total_size;
    float *expected = load_tensor("bins/model_pooled_output.bin", &total_size);
    if (expected) {
        check_tensor(expected, output, 5, "pooler output");
        free(expected);
    }
    free(first_token);
    printf("Pooler forward completed\n");
}

void bert_model_forward(BertModel model, int *input_ids, float *sequence_output, float *pooled_output) {
    printf("Starting model forward\n");
    float *embedding_output = malloc(BATCH_SIZE * SEQ_LENGTH * model.hidden_size * sizeof(float));
    if (!embedding_output) {
        fprintf(stderr, "Failed to allocate embedding output\n");
        exit(1);
    }
    bert_embeddings_forward(model.embeddings, input_ids, embedding_output, model.hidden_size);
    bert_encoder_forward(model.encoder, embedding_output, sequence_output, model.hidden_size, model.intermediate_size);
    bert_pooler_forward(model.pooler, sequence_output, pooled_output, model.hidden_size);
    free(embedding_output);
    printf("Model forward completed\n");
}

int main() {
    printf("Starting main\n");
    BertModel model;
    bert_build_from_checkpoint(&model, "bins/bert_base.bin");

    int total_size;
    int *input_ids = (int *)load_tensor("bins/input_ids.bin", &total_size);
    if (!input_ids) {
        fprintf(stderr, "Failed to load input_ids\n");
        exit(1);
    }
    printf("Input IDs loaded, size=%d\n", total_size);

    float *sequence_output = malloc(BATCH_SIZE * SEQ_LENGTH * model.hidden_size * sizeof(float));
    float *pooled_output = malloc(BATCH_SIZE * model.hidden_size * sizeof(float));
    if (!sequence_output || !pooled_output) {
        fprintf(stderr, "Failed to allocate output buffers\n");
        exit(1);
    }
    printf("Output buffers allocated\n");

    bert_model_forward(model, input_ids, sequence_output, pooled_output);

    float *expected_seq = load_tensor("bins/model_sequence_output.bin", &total_size);
    if (expected_seq) {
        check_tensor(expected_seq, sequence_output, 5, "final sequence output");
        free(expected_seq);
    }

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

    printf("Program completed\n");
    return 0;
}
