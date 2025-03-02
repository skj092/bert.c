#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <stddef.h>

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

// Function to load weights from binary file
void load_weights(const char *filename, float *buffer, size_t size) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    size_t read_size = fread(buffer, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error: Expected %zu elements, but read %zu\n", size, read_size);
        exit(EXIT_FAILURE);
    }

    fclose(file);
}


// Print first few values to verify loading
void print_weights(float *weights, size_t num_values) {
    for (size_t i = 0; i < num_values; i++) {
        printf("%f ", weights[i]);
        if ((i + 1) % 8 == 0) printf("\n");
    }
    printf("\n");
}

void load_int_array(const char *filename, int *buffer, size_t size) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    fread(buffer, sizeof(int), size, file);
    fclose(file);
}

void print_int_array(int *weights, size_t num_values) {
    for (size_t i = 0; i < num_values; i++) {
        printf("%d ", weights[i]);
        if ((i + 1) % 8 == 0) printf("\n");
    }
    printf("\n");
}
