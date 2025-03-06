import torch
import torch.nn as nn
import math
import numpy as np
import os


def write(tensor, handle):
    handle.write(tensor.detach().numpy().astype("float32").tobytes())


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


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


qw_path = "bins/weights/encoder_layer_0_attention_self_query_weight.bin"
kw_path = "bins/weights/encoder_layer_0_attention_self_key_weight.bin"
vw_path = "bins/weights/encoder_layer_0_attention_self_value_weight.bin"
qb_path = "bins/weights/encoder_layer_0_attention_self_query_bias.bin"
kb_path = "bins/weights/encoder_layer_0_attention_self_key_bias.bin"
vb_path = "bins/weights/encoder_layer_0_attention_self_value_bias.bin"


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, dropout=0.1, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5  # Scaling factor for attention scores

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        # load weight for verificiation
        # Load weights and biases from disk
        self.query.weight.data = torch.from_numpy(np.fromfile(qw_path, dtype=np.float32)).view(hidden_size, hidden_size)
        self.key.weight.data = torch.from_numpy(np.fromfile(kw_path, dtype=np.float32)).view(hidden_size, hidden_size)
        self.value.weight.data = torch.from_numpy(np.fromfile(vw_path, dtype=np.float32)).view(hidden_size, hidden_size)

        # Load bias values from disk (assuming the bias vectors are of shape (hidden_size,))
        self.query.bias.data = torch.from_numpy(np.fromfile(qb_path, dtype=np.float32)).view(hidden_size)
        self.key.bias.data = torch.from_numpy(np.fromfile(kb_path, dtype=np.float32)).view(hidden_size)
        self.value.bias.data = torch.from_numpy(np.fromfile(vb_path, dtype=np.float32)).view(hidden_size)
        save_tensor_as_bin("bins/tmp/qw.bin", self.query.weight)
        save_tensor_as_bin("bins/tmp/qb.bin", self.query.bias)
        save_tensor_as_bin("bins/tmp/kw.bin", self.key.weight)
        save_tensor_as_bin("bins/tmp/kb.bin", self.key.bias)
        save_tensor_as_bin("bins/tmp/vw.bin", self.value.weight)
        save_tensor_as_bin("bins/tmp/vb.bin", self.value.bias)
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


def generate_random_input(batch_size=2, seq_length=128, vocab_size=30522):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    return input_ids, attention_mask, token_type_ids


if __name__ == "__main__":
    B, T, C = 2, 128, 768
    x_path = "bins/layer0_attn_input.bin"
    x = torch.from_numpy(np.fromfile(x_path, dtype=np.float32)).view(B, T, C)

    print(f"shape of x: {x.shape}")
    att = BertSelfAttention()
    out = att(x)[0]
    o_path = "bins/att_out.bin"
    with open(o_path, 'wb') as file:
        write(out, file)
    print(out)
    o = torch.from_numpy(np.fromfile(o_path, dtype=np.float32)).view(B, T, C)
