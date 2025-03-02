#include "bert/utils.h"
#include <gtest/gtest.h>
#include <stdlib.h>
#include <string.h>

// Test initialization of BERT embeddings
TEST(BertEmbeddingsTest, Initialization) {
  BertConfig config = default_config();
  BertEmbeddings embeddings = init_bert_embeddings(&config);

  // Ensure memory is allocated
  ASSERT_NE(embeddings.word_embeddings, nullptr);
  ASSERT_NE(embeddings.position_embeddings, nullptr);
  ASSERT_NE(embeddings.token_type_embeddings, nullptr);
  ASSERT_NE(embeddings.layer_norm_weight, nullptr);
  ASSERT_NE(embeddings.layer_norm_bias, nullptr);

  // Ensure layer norm weights are initialized correctly
  for (int i = 0; i < config.hidden_size; i++) {
    EXPECT_FLOAT_EQ(embeddings.layer_norm_weight[i], 1.0f);
    EXPECT_FLOAT_EQ(embeddings.layer_norm_bias[i], 0.0f);
  }
}

// Test word embedding forward pass
TEST(BertEmbeddingsTest, WordEmbeddingsForward) {
  BertConfig config = default_config();
  BertEmbeddings embeddings = init_bert_embeddings(&config);

  int batch_size = 1;
  int seq_length = 4;
  int hidden_size = config.hidden_size;
  int input_ids[] = {100, 200, 300, 400};

  float *output =
      (float *)malloc(batch_size * seq_length * hidden_size * sizeof(float));
  bert_word_embeddings_forward(&embeddings, input_ids, batch_size, seq_length,
                               output);

  // Ensure output is not all zeros (assuming nonzero embeddings)
  bool nonzero_found = false;
  for (int i = 0; i < batch_size * seq_length * hidden_size; i++) {
    if (output[i] != 0.0f) {
      nonzero_found = true;
      break;
    }
  }
  ASSERT_TRUE(nonzero_found);

  free(output);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

