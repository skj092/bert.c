#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include "utils.c"

// Function provided by you
void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }
                    preatt_bth[t2] = val;
                }
                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }
                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}


// Prepares QKV input for the attention_forward function
void prepare_qkv_input(float* x, float* qkv_weights, float* qkv_input,
                       int batch_size, int seq_len, int hidden_size, int num_heads) {
    int head_size = hidden_size / num_heads;

    // Create random weight matrices for Q, K, V projections
    float* query_weight = qkv_weights;
    float* key_weight = qkv_weights + hidden_size * hidden_size;
    float* value_weight = qkv_weights + 2 * hidden_size * hidden_size;

    // Initialize with fixed seed for reproducibility
    init_weights(query_weight, hidden_size * hidden_size, 42);
    init_weights(key_weight, hidden_size * hidden_size, 43);
    init_weights(value_weight, hidden_size * hidden_size, 44);

    // Temporary storage for projections
    float* query_proj = allocate_tensor(batch_size * seq_len * hidden_size);
    float* key_proj = allocate_tensor(batch_size * seq_len * hidden_size);
    float* value_proj = allocate_tensor(batch_size * seq_len * hidden_size);

    // Project input to query, key, value
    linear_layer(x, query_weight, query_proj, batch_size, seq_len, hidden_size, hidden_size);
    linear_layer(x, key_weight, key_proj, batch_size, seq_len, hidden_size, hidden_size);
    linear_layer(x, value_weight, value_proj, batch_size, seq_len, hidden_size, hidden_size);

    // Reorganize into the expected format for attention_forward
    // attention_forward expects input to be (B, T, 3*C) with QKV interleaved per head
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            for (int h = 0; h < num_heads; h++) {
                for (int i = 0; i < head_size; i++) {
                    // Copy query values
                    int qkv_idx = b * seq_len * hidden_size * 3 + t * hidden_size * 3 + h * head_size + i;
                    int q_idx = b * seq_len * hidden_size + t * hidden_size + h * head_size + i;
                    qkv_input[qkv_idx] = query_proj[q_idx];

                    // Copy key values
                    qkv_input[qkv_idx + hidden_size] = key_proj[q_idx];

                    // Copy value values
                    qkv_input[qkv_idx + 2 * hidden_size] = value_proj[q_idx];
                }
            }
        }
    }

    // Free temporary storage
    free(query_proj);
    free(key_proj);
    free(value_proj);
}
// Function to read a shape file and get dimensions
void read_shape_file(const char* filename, int* shape, int* ndim) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Could not open shape file: %s\n", filename);
        exit(1);
    }

    char line[256];
    fgets(line, sizeof(line), file);
    fclose(file);

    // Parse shape string like "(768, 768)"
    char* token = strtok(line, "(,)");
    *ndim = 0;

    while (token != NULL && *ndim < 10) {  // Limit to avoid overflow
        shape[*ndim] = atoi(token);
        (*ndim)++;
        token = strtok(NULL, "(,) ");
    }
}

// Function to load the QKV weights from saved files
void load_qkv_weights(float* qkv_weights, float* qkv_bias, int hidden_size) {
    // Paths to weight files
    const char* query_weight_path = "bins/tmp/qw.bin";
    const char* key_weight_path = "bins/tmp/kw.bin";
    const char* value_weight_path = "bins/tmp/vw.bin";

    // Paths to bias files
    const char* query_bias_path = "bins/tmp/qb.bin";
    const char* key_bias_path = "bins/tmp/kb.bin";
    const char* value_bias_path = "bins/tmp/vb.bin";

    // Paths to shape files
    const char* query_weight_shape_path = "bins/weights/encoder_layer_0_attention_self_query_weight.bin.shape";
    const char* key_weight_shape_path = "bins/weights/encoder_layer_0_attention_self_key_weight.bin.shape";
    const char* value_weight_shape_path = "bins/weights/encoder_layer_0_attention_self_value_weight.bin.shape";
    const char* query_bias_shape_path = "bins/weights/encoder_layer_0_attention_self_query_bias.bin.shape";
    const char* key_bias_shape_path = "bins/weights/encoder_layer_0_attention_self_key_bias.bin.shape";
    const char* value_bias_shape_path = "bins/weights/encoder_layer_0_attention_self_value_bias.bin.shape";

    // Read shapes to verify dimensions
    int weight_shape[10], weight_ndim;
    int bias_shape[10], bias_ndim;

    // Verify query weight dimensions
    read_shape_file(query_weight_shape_path, weight_shape, &weight_ndim);
    if (weight_ndim != 2 || weight_shape[0] != hidden_size || weight_shape[1] != hidden_size) {
        fprintf(stderr, "Unexpected query weight shape: %d x %d\n", weight_shape[0], weight_shape[1]);
        exit(1);
    }

    // Read weights
    printf("Loading query weights from %s...\n", query_weight_path);
    read_binary_file(query_weight_path, qkv_weights, hidden_size * hidden_size);

    printf("Loading key weights from %s...\n", key_weight_path);
    read_binary_file(key_weight_path, qkv_weights + hidden_size * hidden_size, hidden_size * hidden_size);

    printf("Loading value weights from %s...\n", value_weight_path);
    read_binary_file(value_weight_path, qkv_weights + 2 * hidden_size * hidden_size, hidden_size * hidden_size);

    // Read bias shapes
    read_shape_file(query_bias_shape_path, bias_shape, &bias_ndim);
    if (bias_ndim != 1 || bias_shape[0] != hidden_size) {
        fprintf(stderr, "Unexpected query bias shape: %d\n", bias_shape[0]);
        exit(1);
    }

    // Read biases
    printf("Loading query bias from %s...\n", query_bias_path);
    read_binary_file(query_bias_path, qkv_bias, hidden_size);

    printf("Loading key bias from %s...\n", key_bias_path);
    read_binary_file(key_bias_path, qkv_bias + hidden_size, hidden_size);

    printf("Loading value bias from %s...\n", value_bias_path);
    read_binary_file(value_bias_path, qkv_bias + 2 * hidden_size, hidden_size);
}

// Enhanced linear layer with bias
void linear_layer_with_bias(float* input, float* weight, float* bias, float* output,
                           int batch_size, int seq_len, int in_features, int out_features) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int in_offset = b * seq_len * in_features + s * in_features;
            int out_offset = b * seq_len * out_features + s * out_features;

            for (int i = 0; i < out_features; i++) {
                // Initialize with bias
                output[out_offset + i] = bias[i];

                // Add weighted sum
                for (int j = 0; j < in_features; j++) {
                    output[out_offset + i] += input[in_offset + j] * weight[j * out_features + i];
                }
            }
        }
    }
}

// Updated function to prepare QKV input using loaded weights and biases
void prepare_qkv_input_from_saved_weights(float* x, float* qkv_weights, float* qkv_bias, float* qkv_input,
                                         int batch_size, int seq_len, int hidden_size, int num_heads) {
    int head_size = hidden_size / num_heads;

    // Get pointers to the individual weight matrices
    float* query_weight = qkv_weights;
    float* key_weight = qkv_weights + hidden_size * hidden_size;
    float* value_weight = qkv_weights + 2 * hidden_size * hidden_size;

    // Get pointers to the bias vectors
    float* query_bias = qkv_bias;
    float* key_bias = qkv_bias + hidden_size;
    float* value_bias = qkv_bias + 2 * hidden_size;

    // Temporary storage for projections
    float* query_proj = allocate_tensor(batch_size * seq_len * hidden_size);
    float* key_proj = allocate_tensor(batch_size * seq_len * hidden_size);
    float* value_proj = allocate_tensor(batch_size * seq_len * hidden_size);

    // Project input to query, key, value with bias
    linear_layer_with_bias(x, query_weight, query_bias, query_proj,
                           batch_size, seq_len, hidden_size, hidden_size);
    linear_layer_with_bias(x, key_weight, key_bias, key_proj,
                           batch_size, seq_len, hidden_size, hidden_size);
    linear_layer_with_bias(x, value_weight, value_bias, value_proj,
                           batch_size, seq_len, hidden_size, hidden_size);

    // Reorganize into the expected format for attention_forward
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            for (int h = 0; h < num_heads; h++) {
                for (int i = 0; i < head_size; i++) {
                    // Copy query values
                    int qkv_idx = b * seq_len * hidden_size * 3 + t * hidden_size * 3 + h * head_size + i;
                    int q_idx = b * seq_len * hidden_size + t * hidden_size + h * head_size + i;
                    qkv_input[qkv_idx] = query_proj[q_idx];

                    // Copy key values
                    qkv_input[qkv_idx + hidden_size] = key_proj[q_idx];

                    // Copy value values
                    qkv_input[qkv_idx + 2 * hidden_size] = value_proj[q_idx];
                }
            }
        }
    }

    // Free temporary storage
    free(query_proj);
    free(key_proj);
    free(value_proj);
}

int main() {
    // Define model parameters
    const int BATCH_SIZE = 2;
    const int SEQ_LENGTH = 128;
    const int HIDDEN_SIZE = 768;
    const int NUM_HEADS = 12;
    const int HEAD_DIM = HIDDEN_SIZE / NUM_HEADS;

    printf("Starting BERT attention verification...\n");
    printf("Parameters: B=%d, T=%d, C=%d, NH=%d, HS=%d\n",
           BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE, NUM_HEADS, HEAD_DIM);

    // Allocate memory for input tensor
    float* input = (float*)malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * sizeof(float));


    // Read input from binary file
    printf("Reading input from bins/temp.bin...\n");
    read_binary_file("bins/temp.bin", input, BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE);

    // Allocate memory for QKV weights and biases
    float* qkv_weights = allocate_tensor(3 * HIDDEN_SIZE * HIDDEN_SIZE);
    float* qkv_bias = allocate_tensor(3 * HIDDEN_SIZE);

    // Load QKV weights and biases from saved files
    printf("Loading QKV weights and biases from saved files...\n");
    load_qkv_weights(qkv_weights, qkv_bias, HIDDEN_SIZE);

    // // // Allocate memory for QKV input format
    float* qkv_input = (float*)malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE * 3 * sizeof(float));
    //
    // // Prepare QKV input by applying projections with loaded weights
    printf("Preparing QKV input using saved weights...\n");
    prepare_qkv_input_from_saved_weights(input, qkv_weights, qkv_bias, qkv_input,
                                        BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE, NUM_HEADS);

    // Allocate memory for attention intermediate states and output
    float* preatt = allocate_tensor(BATCH_SIZE * NUM_HEADS * SEQ_LENGTH * SEQ_LENGTH);
    float* att = allocate_tensor(BATCH_SIZE * NUM_HEADS * SEQ_LENGTH * SEQ_LENGTH);
    float* output = allocate_tensor(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE);
    //
    // // Run attention forward pass
    printf("Running attention_forward...\n");
    attention_forward(output, preatt, att, qkv_input,
                     BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE, NUM_HEADS);

    // // Write output to file for verification
    // printf("Writing output to bins/c_att_out.bin...\n");
    // write_binary_file("bins/c_att_out.bin", output, BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE);
    //
    // // Load PyTorch output for comparison
    // float* pytorch_output = allocate_tensor(BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE);
    // printf("Reading PyTorch output from bins/att_out.bin...\n");
    // read_binary_file("bins/att_out.bin", pytorch_output, BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE);
    //
    // // Compare outputs
    // float mse = compute_mse(output, pytorch_output, BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE);
    // printf("Mean Squared Error between PyTorch and C implementations: %e\n", mse);
    //
    // // Validation threshold - note that this should be tuned based on the expected accuracy
    // float validation_threshold = 1e-4;
    // if (mse < validation_threshold) {
    //     printf("Verification successful! The C implementation matches the PyTorch output.\n");
    // } else {
    //     printf("Verification failed. There are significant differences between implementations.\n");
    //
    //     // Print a sample of differences for debugging
    //     printf("\nSample differences (first 10 values):\n");
    //     printf("Index\tPyTorch\t\tC\t\tDiff\n");
    //     for (int i = 0; i < 10; i++) {
    //         printf("%d\t%f\t%f\t%f\n",
    //                i, pytorch_output[i], output[i], pytorch_output[i] - output[i]);
    //     }
    //
    //     // Print some stats to help debug
    //     float max_diff = 0.0f;
    //     int max_diff_idx = 0;
    //     for (int i = 0; i < BATCH_SIZE * SEQ_LENGTH * HIDDEN_SIZE; i++) {
    //         float diff = fabsf(pytorch_output[i] - output[i]);
    //         if (diff > max_diff) {
    //             max_diff = diff;
    //             max_diff_idx = i;
    //         }
    //     }
    //     printf("\nMaximum absolute difference: %f at index %d\n", max_diff, max_diff_idx);
    //
    //     // Convert linear index to (batch, seq, hidden) coordinates
    //     int b = max_diff_idx / (SEQ_LENGTH * HIDDEN_SIZE);
    //     int remainder = max_diff_idx % (SEQ_LENGTH * HIDDEN_SIZE);
    //     int t = remainder / HIDDEN_SIZE;
    //     int h = remainder % HIDDEN_SIZE;
    //     printf("This corresponds to position (batch=%d, seq=%d, hidden=%d)\n", b, t, h);
    // }
    //
    // Free allocated memory
    free(input);
    free(qkv_weights);
    free(qkv_bias);
    free(qkv_input);
    free(preatt);
    free(att);
    free(output);
    // free(pytorch_output);

    return 0;
}
