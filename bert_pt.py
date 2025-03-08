"""Modified BERT implementation that saves intermediate outputs for C code validation"""

from transformers import BertModel
import torch
from dataclasses import dataclass
import numpy as np
import torch.nn as nn
import math
import os


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# Ensure output directory exists
os.makedirs("bins", exist_ok=True)


@dataclass
class BertConfig:
    is_decoder: bool = False
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    type_vocab_size: int = 2


def write(tensor, handle):
    handle.write(tensor.detach().numpy().astype("float32").tobytes())


def save_tensor_as_bin(filename, tensor):
    """Save tensor to binary file with proper format for C consumption"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save as float32 (C's float) for most tensors
    if tensor.dtype in [torch.float32, torch.float]:
        tensor.detach().cpu().numpy().astype(np.float32).tofile(filename)
    # Save as int32 for integer tensors
    elif tensor.dtype in [torch.int32, torch.int64, torch.long, torch.int]:
        tensor.detach().cpu().numpy().astype(np.int32).tofile(filename)
    else:
        print(f"Warning: Unhandled dtype {tensor.dtype} for {filename}")
        tensor.detach().cpu().numpy().tofile(filename)

    # Also save shape information for easier loading in C
    with open(f"{filename}.shape", "w") as f:
        f.write(",".join(str(dim) for dim in tensor.shape))


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=0.1)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(self, input_ids, token_type_ids=None):
        batch_size, seq_length = input_ids.size()

        # Word Embeddings
        word_embed = self.word_embeddings(input_ids)
        save_tensor_as_bin("bins/word_embeddings_output.bin", word_embed)

        # Position Embeddings
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)
        save_tensor_as_bin("bins/position_ids.bin", position_ids)

        position_embd = self.position_embeddings(position_ids)
        save_tensor_as_bin(
            "bins/position_embeddings_output.bin", position_embd)

        # Token Type Embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        save_tensor_as_bin("bins/token_type_ids.bin", token_type_ids)

        token_type_embd = self.token_type_embeddings(token_type_ids)
        save_tensor_as_bin(
            "bins/token_type_embeddings_output.bin", token_type_embd)

        # Combine all embeddings
        embeddings = word_embed + position_embd + token_type_embd
        save_tensor_as_bin("bins/combined_embeddings.bin", embeddings)

        # Apply LayerNorm and Dropout
        with open('bins/ln_i.bin', 'wb') as file:
            write(embeddings, file)
        embeddings = self.LayerNorm(embeddings)
        with open('bins/ln_w.bin', 'wb') as file:
            write(self.LayerNorm.weight, file)
        with open('bins/ln_b.bin', 'wb') as file:
            write(self.LayerNorm.bias, file)

        with open('bins/ln_o.bin', 'wb') as file:
            write(embeddings, file)
        # Set eval mode to remove randomness in dropout
        if not self.training:
            embeddings = self.dropout(embeddings)
            save_tensor_as_bin("bins/dropout_output.bin", embeddings)

        return embeddings


def generate_random_input(batch_size=2, seq_length=128, vocab_size=30522):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    return input_ids, attention_mask, token_type_ids


class BertSelfAttention(nn.Module):
    def __init__(self, config, hidden_size=768, num_heads=12, dropout=0.1, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5  # Scaling factor for attention scores

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(
        self, query, key, value, dropout_p=0.0
    ) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        # Save intermediate values
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_attn_scale_factor.bin", torch.tensor([scale_factor]))

        # Matrix multiplication of query and key
        attn_weight = query @ key.transpose(-2, -1)
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_attn_qk_product.bin", attn_weight)

        # Apply scaling
        attn_weight = attn_weight * scale_factor
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_attn_scaled.bin", attn_weight)

        # Add bias (zeros in this case)
        attn_weight += attn_bias

        # Apply softmax
        attn_weight = torch.softmax(attn_weight, dim=-1)
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_attn_softmax.bin", attn_weight)

        # Apply dropout (in eval mode, this is identity)
        if not self.training:
            attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
            save_tensor_as_bin(
                f"bins/layer{self.layer_idx}_attn_dropout.bin", attn_weight)

        # Matrix multiplication with value
        output = attn_weight @ value
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_attn_output.bin", output)

        return output

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.shape

        # Save input to this layer
        save_tensor_as_bin(f"bins/layer{self.layer_idx}_attn_input.bin", x)

        # Project query, key, value
        q_proj = self.query(x)
        save_tensor_as_bin(f"bins/layer{self.layer_idx}_q_proj.bin", q_proj)

        k_proj = self.key(x)
        save_tensor_as_bin(f"bins/layer{self.layer_idx}_k_proj.bin", k_proj)

        v_proj = self.value(x)
        save_tensor_as_bin(f"bins/layer{self.layer_idx}_v_proj.bin", v_proj)

        # Reshape for multi-head attention
        q = q_proj.view(batch_size, seq_length, self.num_heads,
                        self.head_dim).transpose(1, 2)
        save_tensor_as_bin(f"bins/layer{self.layer_idx}_q_reshaped.bin", q)

        k = k_proj.view(batch_size, seq_length, self.num_heads,
                        self.head_dim).transpose(1, 2)
        save_tensor_as_bin(f"bins/layer{self.layer_idx}_k_reshaped.bin", k)

        v = v_proj.view(batch_size, seq_length, self.num_heads,
                        self.head_dim).transpose(1, 2)
        save_tensor_as_bin(f"bins/layer{self.layer_idx}_v_reshaped.bin", v)

        # Compute attention scores
        attn_scores = self.scaled_dot_product_attention(q, k, v)
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_attn_scores.bin", attn_scores)

        # Apply attention to values and reshape back
        context = attn_scores.transpose(1, 2).contiguous().view(
            batch_size, seq_length, hidden_size)
        save_tensor_as_bin(f"bins/layer{self.layer_idx}_context.bin", context)

        return (context,)


class BertIntermediate(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, xb):
        # Save input
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_intermediate_input.bin", xb)

        # Linear projection
        xb = self.dense(xb)
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_intermediate_dense.bin", xb)

        # Activation
        out = self.intermediate_act_fn(xb)
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_intermediate_activation.bin", out)

        return out


class BertOutput(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # Save inputs
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_output_hidden_states.bin", hidden_states)
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_output_input_tensor.bin", input_tensor)

        # Linear projection
        hidden_states = self.dense(hidden_states)
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_output_dense.bin", hidden_states)

        # Dropout (in eval mode, this is identity)
        if not self.training:
            hidden_states = self.dropout(hidden_states)
            save_tensor_as_bin(
                f"bins/layer{self.layer_idx}_output_dropout.bin", hidden_states)

        # Residual connection
        residual = hidden_states + input_tensor
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_output_residual.bin", residual)

        # Layer normalization
        normalized = self.LayerNorm(residual)
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_output_layernorm.bin", normalized)

        return normalized


class BertSelfOutput(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        B, T, C = 2, 128, 768
        # Save inputs
        p = 'bins/t1.bin'
        t = 'bins/tw.bin'
        # save_tensor_as_bin(
        #     f"bins/layer{self.layer_idx}_self_output_hidden_states.bin", hidden_states)
        # save_tensor_as_bin(
        #     f"bins/layer{self.layer_idx}_self_output_input_tensor.bin", input_tensor)
        save_tensor_as_bin(p, hidden_states)
        save_tensor_as_bin(t, input_tensor)

        x = torch.from_numpy(np.fromfile(p, dtype=np.float32)).view(B, T, C)
        x = torch.from_numpy(np.fromfile(t, dtype=np.float32)).view(B, T, C)

        # Linear projection
        hidden_states = self.dense(hidden_states)
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_self_output_dense.bin", hidden_states)

        # Dropout (in eval mode, this is identity)
        if not self.training:
            hidden_states = self.dropout(hidden_states)
            save_tensor_as_bin(
                f"bins/layer{self.layer_idx}_self_output_dropout.bin", hidden_states)

        # Residual connection
        residual = hidden_states + input_tensor
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_self_output_residual.bin", residual)

        # Layer normalization
        normalized = self.LayerNorm(residual)
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_self_output_layernorm.bin", normalized)

        return normalized


class BertAttention(nn.Module):
    def __init__(self, config, hidden_size=768, num_heads=12, dropout=0.1, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.self = BertSelfAttention(
            config, hidden_size, num_heads, dropout, layer_idx)
        self.output = BertSelfOutput(config, layer_idx)

    def forward(self, x):
        # Save input
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_attention_input.bin", x)

        # Self-attention
        self_outputs = self.self(x)


        # Self-output processing
        attention_output = self.output(self_outputs[0], x)
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_attention_output.bin", attention_output)

        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertLayer(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = BertAttention(config, config.hidden_size, config.num_attention_heads,
                                       config.attention_probs_dropout_prob, layer_idx)
        self.intermediate = BertIntermediate(config, layer_idx)
        self.output = BertOutput(config, layer_idx)

    def forward(self, hidden_state):
        # Save input to this layer
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_input.bin", hidden_state)

        # Attention block
        attn_out = self.attention(hidden_state)
        extra_outputs = attn_out[1:]

        # FFN blocks
        intermediate = self.intermediate(attn_out[0])
        layer_output = self.output(intermediate, attn_out[0])

        # Save final layer output
        save_tensor_as_bin(
            f"bins/layer{self.layer_idx}_output.bin", layer_output)

        outputs = (layer_output,) + extra_outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config, i)
                                   for i in range(config.num_hidden_layers)])

    def forward(self, hidden_state):
        # Save initial input to encoder
        save_tensor_as_bin("bins/encoder_input.bin", hidden_state)

        # Process through each layer
        for i, l in enumerate(self.layer):
            layer_output = l(hidden_state)[0]
            hidden_state = layer_output

        # Save final encoder output
        save_tensor_as_bin("bins/encoder_output.bin", hidden_state)

        return hidden_state


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Save input
        save_tensor_as_bin("bins/pooler_input.bin", hidden_states)

        # Extract first token
        first_token_tensor = hidden_states[:, 0]
        save_tensor_as_bin("bins/pooler_first_token.bin", first_token_tensor)

        # Linear projection
        pooled_output = self.dense(first_token_tensor)
        save_tensor_as_bin("bins/pooler_dense.bin", pooled_output)

        # Activation
        pooled_output = self.activation(pooled_output)
        save_tensor_as_bin("bins/pooler_activation.bin", pooled_output)

        return pooled_output


class BertModelCustom(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def load_from_pretrained(self, bert_base):
        hf_sd = bert_base.state_dict()

        # Save model weights for C implementation
        for k, v in hf_sd.items():
            save_tensor_as_bin(f"bins/weights/{k.replace('.', '_')}.bin", v)

        # Copy weights to our model
        for k in hf_sd.keys():
            if k in self.state_dict():
                self.state_dict()[k].copy_(hf_sd[k])
            else:
                print(f"Warning: Key {k} not found in model")

        return self

    def forward(self, input_ids, token_type_ids=None, output_all_encoded_layers=False):
        # Save input
        save_tensor_as_bin("bins/model_input_ids.bin", input_ids)
        if token_type_ids is not None:
            save_tensor_as_bin("bins/model_token_type_ids.bin", token_type_ids)

        # Get embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)
        save_tensor_as_bin("bins/model_embedding_output.bin", embedding_output)

        # Process through encoder
        sequence_output = self.encoder(embedding_output)
        save_tensor_as_bin("bins/model_sequence_output.bin", sequence_output)

        # Get pooled output
        pooled_output = self.pooler(sequence_output)
        save_tensor_as_bin("bins/model_pooled_output.bin", pooled_output)

        return sequence_output, pooled_output


def save_model_config(config):
    """Save model configuration as a text file for C implementation"""
    config_dict = {k: v for k, v in config.__dict__.items()}
    with open("bins/model_config.txt", "w") as f:
        for key, value in config_dict.items():
            f.write(f"{key}={value}\n")


if __name__ == "__main__":
    # Create config and save it
    config = BertConfig()
    save_model_config(config)

    # Generate inputs
    input_ids, attention_mask, token_type_ids = generate_random_input()

    # Save inputs
    save_tensor_as_bin("bins/input_ids.bin", input_ids)
    save_tensor_as_bin("bins/attention_mask.bin", attention_mask)
    save_tensor_as_bin("bins/token_type_ids.bin", token_type_ids)

    # Load base BERT model
    bert_base = BertModel.from_pretrained("bert-base-uncased")
    bert_base.eval()  # Set to evaluation mode to disable dropout

    # Create custom model and load weights
    model = BertModelCustom(config)
    model.load_from_pretrained(bert_base)
    model.eval()  # Set to evaluation mode to disable dropout

    # Generate outputs from both models for comparison
    with torch.no_grad():
        # Original model output
        out1 = bert_base(input_ids)
        save_tensor_as_bin(
            "bins/original_last_hidden_state.bin", out1.last_hidden_state)
        save_tensor_as_bin(
            "bins/original_pooler_output.bin", out1.pooler_output)

        # Custom model output
        sequence_output, pooled_output = model(input_ids)
        breakpoint()

        # Verify outputs match
        if torch.allclose(out1.last_hidden_state, sequence_output, atol=1e-5):
            print("✅ Last hidden state outputs match!")
        else:
            print("❌ Last hidden state outputs mismatch!")

        if torch.allclose(out1.pooler_output, pooled_output, atol=1e-5):
            print("✅ Pooler outputs match!")
        else:
            print("❌ Pooler outputs mismatch!")
