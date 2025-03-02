#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <stddef.h>

// // Define implementation for is_training() from bert_utils.h
// bool is_training() {
//     // This would be controlled by a global flag or parameter in a real implementation
//     return false;
// }
// Configuration structure for BERT model
typedef struct {
    bool is_decoder;
    int vocab_size;
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int intermediate_size;
    char hidden_act[10];
    float hidden_dropout_prob;
    float attention_probs_dropout_prob;
    int max_position_embeddings;
    float layer_norm_eps;
    int pad_token_id;
    int type_vocab_size;
} BertConfig;

// Default configuration for BERT base
BertConfig default_config() {
    BertConfig config;
    config.is_decoder = false;
    config.vocab_size = 30522;
    config.hidden_size = 768;
    config.num_hidden_layers = 12;
    config.num_attention_heads = 12;
    config.intermediate_size = 3072;
    strcpy(config.hidden_act, "gelu");
    config.hidden_dropout_prob = 0.1;
    config.attention_probs_dropout_prob = 0.1;
    config.max_position_embeddings = 512;
    config.layer_norm_eps = 1e-12;
    config.pad_token_id = 0;
    config.type_vocab_size = 2;
    return config;
}

// BERT Embeddings component
typedef struct {
    float *word_embeddings;    // [vocab_size, hidden_size]
    float *position_embeddings; // [max_position_embeddings, hidden_size]
    float *token_type_embeddings; // [type_vocab_size, hidden_size]
    float *layer_norm_weight;   // [hidden_size]
    float *layer_norm_bias;     // [hidden_size]
    BertConfig *config;
} BertEmbeddings;

// Self-attention component
typedef struct {
    float *query_weight;  // [hidden_size, hidden_size]
    float *query_bias;    // [hidden_size]
    float *key_weight;    // [hidden_size, hidden_size]
    float *key_bias;      // [hidden_size]
    float *value_weight;  // [hidden_size, hidden_size]
    float *value_bias;    // [hidden_size]
    int num_heads;
    int head_dim;
    float scale;
} BertSelfAttention;

// Self-output component for attention
typedef struct {
    float *dense_weight;  // [hidden_size, hidden_size]
    float *dense_bias;    // [hidden_size]
    float *layer_norm_weight;  // [hidden_size]
    float *layer_norm_bias;    // [hidden_size]
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
    float *dense_weight;  // [hidden_size, intermediate_size]
    float *dense_bias;    // [intermediate_size]
} BertIntermediate;

// Output component
typedef struct {
    float *dense_weight;  // [intermediate_size, hidden_size]
    float *dense_bias;    // [hidden_size]
    float *layer_norm_weight;  // [hidden_size]
    float *layer_norm_bias;    // [hidden_size]
    float dropout_prob;
    float layer_norm_eps;
} BertOutput;

// Layer component
typedef struct {
    BertAttention attention;
    BertIntermediate intermediate;
    BertOutput output;
} BertLayer;

// Encoder component
typedef struct {
    BertLayer *layers;  // Array of num_hidden_layers layers
    int num_layers;
} BertEncoder;

// Pooler component
typedef struct {
    float *dense_weight;  // [hidden_size, hidden_size]
    float *dense_bias;    // [hidden_size]
} BertPooler;

// Complete BERT model
typedef struct {
    BertConfig config;
    BertEmbeddings embeddings;
    BertEncoder encoder;
    BertPooler pooler;
} BertModel;

// Forward declarations
void bert_embeddings_forward(BertEmbeddings *embeddings, int *input_ids, int *token_type_ids,
                            int batch_size, int seq_length, float *output);
void bert_encoder_forward(BertEncoder *encoder, float *hidden_states, int batch_size, int seq_length,
                         int hidden_size, float *output);
void bert_pooler_forward(BertPooler *pooler, float *hidden_states, int batch_size, int hidden_size,
                        float *output);

// Initialize BERT embeddings
BertEmbeddings init_bert_embeddings(BertConfig *config) {
    BertEmbeddings embeddings;
    embeddings.config = config;

    // Allocate memory for embeddings
    size_t word_emb_size = config->vocab_size * config->hidden_size * sizeof(float);
    size_t pos_emb_size = config->max_position_embeddings * config->hidden_size * sizeof(float);
    size_t type_emb_size = config->type_vocab_size * config->hidden_size * sizeof(float);
    size_t norm_size = config->hidden_size * sizeof(float);

    embeddings.word_embeddings = (float*)malloc(word_emb_size);
    embeddings.position_embeddings = (float*)malloc(pos_emb_size);
    embeddings.token_type_embeddings = (float*)malloc(type_emb_size);
    embeddings.layer_norm_weight = (float*)malloc(norm_size);
    embeddings.layer_norm_bias = (float*)malloc(norm_size);

    // Initialize with zeros (in practice, these would be loaded from a pre-trained model)
    memset(embeddings.word_embeddings, 0, word_emb_size);
    memset(embeddings.position_embeddings, 0, pos_emb_size);
    memset(embeddings.token_type_embeddings, 0, type_emb_size);

    // Initialize layer norm weights to 1 and biases to 0
    for (int i = 0; i < config->hidden_size; i++) {
        embeddings.layer_norm_weight[i] = 1.0f;
        embeddings.layer_norm_bias[i] = 0.0f;
    }

    return embeddings;
}

// Initialize self-attention
BertSelfAttention init_bert_self_attention(int hidden_size, int num_heads) {
    BertSelfAttention self_attn;
    self_attn.num_heads = num_heads;
    self_attn.head_dim = hidden_size / num_heads;
    self_attn.scale = 1.0f / sqrt(self_attn.head_dim);

    // Allocate memory for weights and biases
    size_t weight_size = hidden_size * hidden_size * sizeof(float);
    size_t bias_size = hidden_size * sizeof(float);

    self_attn.query_weight = (float*)malloc(weight_size);
    self_attn.query_bias = (float*)malloc(bias_size);
    self_attn.key_weight = (float*)malloc(weight_size);
    self_attn.key_bias = (float*)malloc(bias_size);
    self_attn.value_weight = (float*)malloc(weight_size);
    self_attn.value_bias = (float*)malloc(bias_size);

    // Initialize with zeros
    memset(self_attn.query_weight, 0, weight_size);
    memset(self_attn.query_bias, 0, bias_size);
    memset(self_attn.key_weight, 0, weight_size);
    memset(self_attn.key_bias, 0, bias_size);
    memset(self_attn.value_weight, 0, weight_size);
    memset(self_attn.value_bias, 0, bias_size);

    return self_attn;
}

// Initialize self-output
BertSelfOutput init_bert_self_output(BertConfig *config) {
    BertSelfOutput self_output;
    self_output.dropout_prob = config->hidden_dropout_prob;
    self_output.layer_norm_eps = config->layer_norm_eps;

    // Allocate memory for weights and biases
    size_t weight_size = config->hidden_size * config->hidden_size * sizeof(float);
    size_t bias_size = config->hidden_size * sizeof(float);

    self_output.dense_weight = (float*)malloc(weight_size);
    self_output.dense_bias = (float*)malloc(bias_size);
    self_output.layer_norm_weight = (float*)malloc(bias_size);
    self_output.layer_norm_bias = (float*)malloc(bias_size);

    // Initialize with zeros
    memset(self_output.dense_weight, 0, weight_size);
    memset(self_output.dense_bias, 0, bias_size);

    // Initialize layer norm weights to 1 and biases to 0
    for (int i = 0; i < config->hidden_size; i++) {
        self_output.layer_norm_weight[i] = 1.0f;
        self_output.layer_norm_bias[i] = 0.0f;
    }

    return self_output;
}

// Initialize attention
BertAttention init_bert_attention(BertConfig *config) {
    BertAttention attention;
    attention.self = init_bert_self_attention(config->hidden_size, config->num_attention_heads);
    attention.output = init_bert_self_output(config);
    return attention;
}

// Initialize intermediate
BertIntermediate init_bert_intermediate(BertConfig *config) {
    BertIntermediate intermediate;

    // Allocate memory for weights and biases
    size_t weight_size = config->hidden_size * config->intermediate_size * sizeof(float);
    size_t bias_size = config->intermediate_size * sizeof(float);

    intermediate.dense_weight = (float*)malloc(weight_size);
    intermediate.dense_bias = (float*)malloc(bias_size);

    // Initialize with zeros
    memset(intermediate.dense_weight, 0, weight_size);
    memset(intermediate.dense_bias, 0, bias_size);

    return intermediate;
}

// Initialize output
BertOutput init_bert_output(BertConfig *config) {
    BertOutput output;
    output.dropout_prob = config->hidden_dropout_prob;
    output.layer_norm_eps = config->layer_norm_eps;

    // Allocate memory for weights and biases
    size_t weight_size = config->intermediate_size * config->hidden_size * sizeof(float);
    size_t bias_size = config->hidden_size * sizeof(float);

    output.dense_weight = (float*)malloc(weight_size);
    output.dense_bias = (float*)malloc(bias_size);
    output.layer_norm_weight = (float*)malloc(bias_size);
    output.layer_norm_bias = (float*)malloc(bias_size);

    // Initialize with zeros
    memset(output.dense_weight, 0, weight_size);
    memset(output.dense_bias, 0, bias_size);

    // Initialize layer norm weights to 1 and biases to 0
    for (int i = 0; i < config->hidden_size; i++) {
        output.layer_norm_weight[i] = 1.0f;
        output.layer_norm_bias[i] = 0.0f;
    }

    return output;
}

// Initialize layer
BertLayer init_bert_layer(BertConfig *config) {
    BertLayer layer;
    layer.attention = init_bert_attention(config);
    layer.intermediate = init_bert_intermediate(config);
    layer.output = init_bert_output(config);
    return layer;
}

// Initialize encoder
BertEncoder init_bert_encoder(BertConfig *config) {
    BertEncoder encoder;
    encoder.num_layers = config->num_hidden_layers;

    // Allocate memory for layers
    encoder.layers = (BertLayer*)malloc(config->num_hidden_layers * sizeof(BertLayer));

    // Initialize each layer
    for (int i = 0; i < config->num_hidden_layers; i++) {
        encoder.layers[i] = init_bert_layer(config);
    }

    return encoder;
}

// Initialize pooler
BertPooler init_bert_pooler(BertConfig *config) {
    BertPooler pooler;

    // Allocate memory for weights and biases
    size_t weight_size = config->hidden_size * config->hidden_size * sizeof(float);
    size_t bias_size = config->hidden_size * sizeof(float);

    pooler.dense_weight = (float*)malloc(weight_size);
    pooler.dense_bias = (float*)malloc(bias_size);

    // Initialize with zeros
    memset(pooler.dense_weight, 0, weight_size);
    memset(pooler.dense_bias, 0, bias_size);

    return pooler;
}

// Initialize BERT model
BertModel* init_bert_model(BertConfig *config) {
    BertModel *model = (BertModel*)malloc(sizeof(BertModel));
    model->config = *config;
    model->embeddings = init_bert_embeddings(config);
    model->encoder = init_bert_encoder(config);
    model->pooler = init_bert_pooler(config);
    return model;
}

// Forward pass for BERT embeddings
void bert_embeddings_forward(BertEmbeddings *embeddings, int *input_ids, int *token_type_ids,
                            int batch_size, int seq_length, float *output) {
    int hidden_size = embeddings->config->hidden_size;

    // Initialize output with zeros
    size_t output_size = batch_size * seq_length * hidden_size * sizeof(float);
    memset(output, 0, output_size);

    // Create temporary buffer for embeddings
    float *temp_embeddings = (float*)malloc(output_size);
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
                temp_embeddings[emb_idx] += embeddings->position_embeddings[pos_emb_idx];

                // Add token type embedding
                int type_emb_idx = type_idx * hidden_size + h;
                temp_embeddings[emb_idx] += embeddings->token_type_embeddings[type_emb_idx];
            }
        }
    }

    // Apply layer normalization
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            float mean = 0.0f;
            float variance = 0.0f;

            // Calculate mean
            for (int h = 0; h < hidden_size; h++) {
                int idx = b * seq_length * hidden_size + s * hidden_size + h;
                mean += temp_embeddings[idx];
            }
            mean /= hidden_size;

            // Calculate variance
            for (int h = 0; h < hidden_size; h++) {
                int idx = b * seq_length * hidden_size + s * hidden_size + h;
                float diff = temp_embeddings[idx] - mean;
                variance += diff * diff;
            }
            variance /= hidden_size;

            // Apply layer norm
            for (int h = 0; h < hidden_size; h++) {
                int idx = b * seq_length * hidden_size + s * hidden_size + h;
                float normalized = (temp_embeddings[idx] - mean) / sqrt(variance + embeddings->config->layer_norm_eps);
                output[idx] = embeddings->layer_norm_weight[h] * normalized + embeddings->layer_norm_bias[h];

                // Apply dropout (simplified: just scaling by dropout probability)
                if (is_training()) {
                    output[idx] *= (1.0f - embeddings->config->hidden_dropout_prob);
                }
            }
        }
    }

    free(temp_embeddings);
}

// Self-attention forward pass
void bert_self_attention_forward(BertSelfAttention *self_attn, float *hidden_states,
                                int batch_size, int seq_length, int hidden_size, float *output) {
    // Allocate memory for Q, K, V tensors and attention scores
    size_t qkv_size = batch_size * self_attn->num_heads * seq_length * self_attn->head_dim * sizeof(float);
    float *query = (float*)malloc(qkv_size);
    float *key = (float*)malloc(qkv_size);
    float *value = (float*)malloc(qkv_size);
    float *attention_scores = (float*)malloc(batch_size * self_attn->num_heads * seq_length * seq_length * sizeof(float));

    // Project query, key, value
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            // Calculate query
            for (int h = 0; h < self_attn->num_heads; h++) {
                for (int d = 0; d < self_attn->head_dim; d++) {
                    float sum = 0.0f;
                    for (int i = 0; i < hidden_size; i++) {
                        int hidden_idx = b * seq_length * hidden_size + s * hidden_size + i;
                        int weight_idx = i * hidden_size + h * self_attn->head_dim + d;
                        sum += hidden_states[hidden_idx] * self_attn->query_weight[weight_idx];
                    }
                    int q_idx = b * self_attn->num_heads * seq_length * self_attn->head_dim +
                               h * seq_length * self_attn->head_dim +
                               s * self_attn->head_dim + d;
                    query[q_idx] = sum + self_attn->query_bias[h * self_attn->head_dim + d];
                }
            }

            // Calculate key and value (similar to query)
            for (int h = 0; h < self_attn->num_heads; h++) {
                for (int d = 0; d < self_attn->head_dim; d++) {
                    float key_sum = 0.0f;
                    float value_sum = 0.0f;
                    for (int i = 0; i < hidden_size; i++) {
                        int hidden_idx = b * seq_length * hidden_size + s * hidden_size + i;
                        int weight_idx = i * hidden_size + h * self_attn->head_dim + d;
                        key_sum += hidden_states[hidden_idx] * self_attn->key_weight[weight_idx];
                        value_sum += hidden_states[hidden_idx] * self_attn->value_weight[weight_idx];
                    }
                    int kv_idx = b * self_attn->num_heads * seq_length * self_attn->head_dim +
                                h * seq_length * self_attn->head_dim +
                                s * self_attn->head_dim + d;
                    key[kv_idx] = key_sum + self_attn->key_bias[h * self_attn->head_dim + d];
                    value[kv_idx] = value_sum + self_attn->value_bias[h * self_attn->head_dim + d];
                }
            }
        }
    }

    // Compute attention scores: Q * K^T / sqrt(head_dim)
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < self_attn->num_heads; h++) {
            for (int q_seq = 0; q_seq < seq_length; q_seq++) {
                for (int k_seq = 0; k_seq < seq_length; k_seq++) {
                    float score = 0.0f;
                    for (int d = 0; d < self_attn->head_dim; d++) {
                        int q_idx = b * self_attn->num_heads * seq_length * self_attn->head_dim +
                                   h * seq_length * self_attn->head_dim +
                                   q_seq * self_attn->head_dim + d;
                        int k_idx = b * self_attn->num_heads * seq_length * self_attn->head_dim +
                                   h * seq_length * self_attn->head_dim +
                                   k_seq * self_attn->head_dim + d;
                        score += query[q_idx] * key[k_idx];
                    }
                    int score_idx = b * self_attn->num_heads * seq_length * seq_length +
                                   h * seq_length * seq_length +
                                   q_seq * seq_length + k_seq;
                    attention_scores[score_idx] = score * self_attn->scale;
                }
            }
        }
    }

    // Apply softmax to attention scores
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < self_attn->num_heads; h++) {
            for (int q_seq = 0; q_seq < seq_length; q_seq++) {
                // Find max for numerical stability
                float max_val = -INFINITY;
                for (int k_seq = 0; k_seq < seq_length; k_seq++) {
                    int score_idx = b * self_attn->num_heads * seq_length * seq_length +
                                   h * seq_length * seq_length +
                                   q_seq * seq_length + k_seq;
                    if (attention_scores[score_idx] > max_val) {
                        max_val = attention_scores[score_idx];
                    }
                }

                // Calculate softmax denominator
                float sum_exp = 0.0f;
                for (int k_seq = 0; k_seq < seq_length; k_seq++) {
                    int score_idx = b * self_attn->num_heads * seq_length * seq_length +
                                   h * seq_length * seq_length +
                                   q_seq * seq_length + k_seq;
                    attention_scores[score_idx] = exp(attention_scores[score_idx] - max_val);
                    sum_exp += attention_scores[score_idx];
                }

                // Normalize
                for (int k_seq = 0; k_seq < seq_length; k_seq++) {
                    int score_idx = b * self_attn->num_heads * seq_length * seq_length +
                                   h * seq_length * seq_length +
                                   q_seq * seq_length + k_seq;
                    attention_scores[score_idx] /= sum_exp;
                }
            }
        }
    }

    // Apply attention weights to values
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < self_attn->num_heads; h++) {
            for (int q_seq = 0; q_seq < seq_length; q_seq++) {
                for (int d = 0; d < self_attn->head_dim; d++) {
                    float weighted_sum = 0.0f;
                    for (int k_seq = 0; k_seq < seq_length; k_seq++) {
                        int score_idx = b * self_attn->num_heads * seq_length * seq_length +
                                       h * seq_length * seq_length +
                                       q_seq * seq_length + k_seq;
                        int v_idx = b * self_attn->num_heads * seq_length * self_attn->head_dim +
                                   h * seq_length * self_attn->head_dim +
                                   k_seq * self_attn->head_dim + d;
                        weighted_sum += attention_scores[score_idx] * value[v_idx];
                    }

                    // Store result in output
                    int out_idx = b * seq_length * hidden_size + q_seq * hidden_size + h * self_attn->head_dim + d;
                    output[out_idx] = weighted_sum;
                }
            }
        }
    }

    // Free memory
    free(query);
    free(key);
    free(value);
    free(attention_scores);
}

// Self-output forward pass
void bert_self_output_forward(BertSelfOutput *self_output, float *hidden_states, float *input_tensor,
                             int batch_size, int seq_length, int hidden_size, float *output) {
    // Temporary buffer for dense output
    float *dense_output = (float*)malloc(batch_size * seq_length * hidden_size * sizeof(float));

    // Apply dense layer
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            for (int h_out = 0; h_out < hidden_size; h_out++) {
                float sum = 0.0f;
                for (int h_in = 0; h_in < hidden_size; h_in++) {
                    int in_idx = b * seq_length * hidden_size + s * hidden_size + h_in;
                    sum += hidden_states[in_idx] * self_output->dense_weight[h_in * hidden_size + h_out];
                }
                int out_idx = b * seq_length * hidden_size + s * hidden_size + h_out;
                dense_output[out_idx] = sum + self_output->dense_bias[h_out];

                // Apply dropout
                if (is_training()) {
                    dense_output[out_idx] *= (1.0f - self_output->dropout_prob);
                }
            }
        }
    }

    // Add residual connection
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            for (int h = 0; h < hidden_size; h++) {
                int idx = b * seq_length * hidden_size + s * hidden_size + h;
                dense_output[idx] += input_tensor[idx];
            }
        }
    }

    // Apply layer normalization
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            float mean = 0.0f;
            float variance = 0.0f;

            // Calculate mean
            for (int h = 0; h < hidden_size; h++) {
                int idx = b * seq_length * hidden_size + s * hidden_size + h;
                mean += dense_output[idx];
            }
            mean /= hidden_size;

            // Calculate variance
            for (int h = 0; h < hidden_size; h++) {
                int idx = b * seq_length * hidden_size + s * hidden_size + h;
                float diff = dense_output[idx] - mean;
                variance += diff * diff;
            }
            variance /= hidden_size;

            // Apply layer norm
            for (int h = 0; h < hidden_size; h++) {
                int idx = b * seq_length * hidden_size + s * hidden_size + h;
                float normalized = (dense_output[idx] - mean) / sqrt(variance + self_output->layer_norm_eps);
                output[idx] = self_output->layer_norm_weight[h] * normalized + self_output->layer_norm_bias[h];
            }
        }
    }

    free(dense_output);
}

// Attention forward pass
void bert_attention_forward(BertAttention *attention, float *hidden_states,
                           int batch_size, int seq_length, int hidden_size, float *output) {
    // Temporary buffer for self attention output
    float *self_output = (float*)malloc(batch_size * seq_length * hidden_size * sizeof(float));

    // Self attention forward
    bert_self_attention_forward(&attention->self, hidden_states, batch_size, seq_length, hidden_size, self_output);

    // Self output forward
    bert_self_output_forward(&attention->output, self_output, hidden_states, batch_size, seq_length, hidden_size, output);

    free(self_output);
}

// GELU activation function
float gelu(float x) {
    return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

// Intermediate forward pass
void bert_intermediate_forward(BertIntermediate *intermediate, float *hidden_states,
                              int batch_size, int seq_length, int hidden_size, int intermediate_size, float *output) {
    // Apply dense layer
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            for (int i = 0; i < intermediate_size; i++) {
                float sum = 0.0f;
                for (int h = 0; h < hidden_size; h++) {
                    int in_idx = b * seq_length * hidden_size + s * hidden_size + h;
                    sum += hidden_states[in_idx] * intermediate->dense_weight[h * intermediate_size + i];
                }
                int out_idx = b * seq_length * intermediate_size + s * intermediate_size + i;
                output[out_idx] = gelu(sum + intermediate->dense_bias[i]);
            }
        }
    }
}

// Output forward pass
void bert_output_forward(BertOutput *output, float *hidden_states, float *attention_output,
                        int batch_size, int seq_length, int hidden_size, int intermediate_size, float *out) {
    // Temporary buffer for dense output
    float *dense_output = (float*)malloc(batch_size * seq_length * hidden_size * sizeof(float));

    // Apply dense layer
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            for (int h = 0; h < hidden_size; h++) {
                float sum = 0.0f;
                for (int i = 0; i < intermediate_size; i++) {
                    int in_idx = b * seq_length * intermediate_size + s * intermediate_size + i;
                    sum += hidden_states[in_idx] * output->dense_weight[i * hidden_size + h];
                }
                int out_idx = b * seq_length * hidden_size + s * hidden_size + h;
                dense_output[out_idx] = sum + output->dense_bias[h];

                // Apply dropout
                if (is_training()) {
                    dense_output[out_idx] *= (1.0f - output->dropout_prob);
                }
            }
        }
    }

    // Add residual connection
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            for (int h = 0; h < hidden_size; h++) {
                int idx = b * seq_length * hidden_size + s * hidden_size + h;
                dense_output[idx] += attention_output[idx];
            }
        }
    }

    // Apply layer normalization
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            float mean = 0.0f;
            float variance = 0.0f;

            // Calculate mean
            for (int h = 0; h < hidden_size; h++) {
                int idx = b * seq_length * hidden_size + s * hidden_size + h;
                mean += dense_output[idx];
            }
            mean /= hidden_size;

            // Calculate variance
            for (int h = 0; h < hidden_size; h++) {
                int idx = b * seq_length * hidden_size + s * hidden_size + h;
                float diff = dense_output[idx] - mean;
                variance += diff * diff;
            }
            variance /= hidden_size;

            // Apply layer norm
            for (int h = 0; h < hidden_size; h++) {
                int idx = b * seq_length * hidden_size + s * hidden_size + h;
                float normalized = (dense_output[idx] - mean) / sqrt(variance + output->layer_norm_eps);
                out[idx] = output->layer_norm_weight[h] * normalized + output->layer_norm_bias[h];
            }
        }
    }

    free(dense_output);
}

// Layer forward pass
void bert_layer_forward(BertLayer *layer, float *hidden_states,
                       int batch_size, int seq_length, int hidden_size, int intermediate_size, float *output) {
    // Temporary buffers
    float *attention_output = (float*)malloc(batch_size * seq_length * hidden_size * sizeof(float));
    float *intermediate_output = (float*)malloc(batch_size * seq_length * intermediate_size * sizeof(float));

    // Attention forward
    bert_attention_forward(&layer->attention, hidden_states, batch_size, seq_length, hidden_size, attention_output);

    // Intermediate forward
    bert_intermediate_forward(&layer->intermediate, attention_output, batch_size, seq_length, hidden_size, intermediate_size, intermediate_output);

    // Output forward
    bert_output_forward(&layer->output, intermediate_output, attention_output, batch_size, seq_length, hidden_size, intermediate_size, output);

    // Free memory
    free(attention_output);
    free(intermediate_output);
}

// Encoder forward pass
void bert_encoder_forward(BertEncoder *encoder, float *hidden_states, int batch_size, int seq_length,
                         int hidden_size, float *output) {
    // Copy input to output for first layer
    memcpy(output, hidden_states, batch_size * seq_length * hidden_size * sizeof(float));

    // Temporary buffer for layer output
    float *layer_output = (float*)malloc(batch_size * seq_length * hidden_size * sizeof(float));
    float *current_input = output;

    // Process each layer
    for (int i = 0; i < encoder->num_layers; i++) {
        BertLayer *layer = &encoder->layers[i];

        // Process layer
        bert_layer_forward(layer, current_input, batch_size, seq_length, hidden_size,
                         ((BertConfig*)layer->intermediate.dense_bias - offsetof(BertConfig, pad_token_id))->intermediate_size,
                         layer_output);

        // Update current input
        memcpy(current_input, layer_output, batch_size * seq_length * hidden_size * sizeof(float));
    }

    free(layer_output);
}

// Pooler forward pass
void bert_pooler_forward(BertPooler *pooler, float *hidden_states, int batch_size, int hidden_size,
                        float *output) {
    // Get first token ([CLS]) hidden states for each batch
    for (int b = 0; b < batch_size; b++) {
        // Apply dense layer
        for (int h_out = 0; h_out < hidden_size; h_out++) {
            float sum = 0.0f;
            for (int h_in = 0; h_in < hidden_size; h_in++) {
                int in_idx = b * hidden_size + h_in;
                sum += hidden_states[in_idx] * pooler->dense_weight[h_in * hidden_size + h_out];
            }
            int out_idx = b * hidden_size + h_out;
            output[out_idx] = tanh(sum + pooler->dense_bias[h_out]);
        }
    }
}

// Main BERT model forward pass
void bert_model_forward(BertModel *model, int *input_ids, int *token_type_ids, int batch_size, int seq_length,
                        float *sequence_output, float *pooled_output) {
    int hidden_size = model->config.hidden_size;

    // Allocate memory for embeddings output
    float *embedding_output = (float*)malloc(batch_size * seq_length * hidden_size * sizeof(float));

    // Embeddings forward
    bert_embeddings_forward(&model->embeddings, input_ids, token_type_ids, batch_size, seq_length, embedding_output);

    // Encoder forward
    bert_encoder_forward(&model->encoder, embedding_output, batch_size, seq_length, hidden_size, sequence_output);

    // Pooler forward
    bert_pooler_forward(&model->pooler, sequence_output, batch_size, hidden_size, pooled_output);

    // Free memory
    free(embedding_output);
}

// Free memory for BERT embeddings
void free_bert_embeddings(BertEmbeddings *embeddings) {
    free(embeddings->word_embeddings);
    free(embeddings->position_embeddings);
    free(embeddings->token_type_embeddings);
    free(embeddings->layer_norm_weight);
    free(embeddings->layer_norm_bias);
}

// Free memory for self-attention
void free_bert_self_attention(BertSelfAttention *self_attn) {
    free(self_attn->query_weight);
    free(self_attn->query_bias);
    free(self_attn->key_weight);
    free(self_attn->key_bias);
    free(self_attn->value_weight);
    free(self_attn->value_bias);
}

// Free memory for self-output
void free_bert_self_output(BertSelfOutput *self_output) {
    free(self_output->dense_weight);
    free(self_output->dense_bias);
    free(self_output->layer_norm_weight);
    free(self_output->layer_norm_bias);
}

// Free memory for attention
void free_bert_attention(BertAttention *attention) {
    free_bert_self_attention(&attention->self);
    free_bert_self_output(&attention->output);
}

// Free memory for intermediate
void free_bert_intermediate(BertIntermediate *intermediate) {
    free(intermediate->dense_weight);
    free(intermediate->dense_bias);
}

// Free memory for output
void free_bert_output(BertOutput *output) {
    free(output->dense_weight);
    free(output->dense_bias);
    free(output->layer_norm_weight);
    free(output->layer_norm_bias);
}

// Free memory for layer
void free_bert_layer(BertLayer *layer) {
    free_bert_attention(&layer->attention);
    free_bert_intermediate(&layer->intermediate);
    free_bert_output(&layer->output);
}

// Free memory for encoder
void free_bert_encoder(BertEncoder *encoder) {
    for (int i = 0; i < encoder->num_layers; i++) {
        free_bert_layer(&encoder->layers[i]);
    }
    free(encoder->layers);
}

// Free memory for pooler
void free_bert_pooler(BertPooler *pooler) {
    free(pooler->dense_weight);
    free(pooler->dense_bias);
}

// Free memory for BERT model
void free_bert_model(BertModel *model) {
    free_bert_embeddings(&model->embeddings);
    free_bert_encoder(&model->encoder);
    free_bert_pooler(&model->pooler);
    free(model);
}

// Example usage
int main() {
    // Initialize configuration
    BertConfig config = default_config();

    // Initialize model
    BertModel *model = init_bert_model(&config);

    // Sample input
    int batch_size = 2;
    int seq_length = 8;
    int input_ids[16] = {101, 2054, 2003, 2026, 2171, 1012, 102, 0,  // "What is my name?"
                         101, 2129, 2003, 2026, 2171, 1029, 102, 0}; // "Who is my name?"
    int token_type_ids[16] = {0};

    // Allocate memory for output
    float *sequence_output = (float*)malloc(batch_size * seq_length * config.hidden_size * sizeof(float));
    float *pooled_output = (float*)malloc(batch_size * config.hidden_size * sizeof(float));

    // Forward pass
    bert_model_forward(model, input_ids, token_type_ids, batch_size, seq_length, sequence_output, pooled_output);

    // Print sample output
    printf("Sample sequence output for first token of first batch:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", sequence_output[i]);
    }
    printf("...\n");

    printf("Sample pooled output for first batch:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", pooled_output[i]);
    }
    printf("...\n");

    // Free memory
    free(sequence_output);
    free(pooled_output);
    free_bert_model(model);

    return 0;
}

