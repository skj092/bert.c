from dataclasses import dataclass
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


# def save_tensor_as_bin(filename, tensor):
#     """Save tensor to binary file with proper format for C consumption"""
#     # Create directory if it doesn't exist
#     os.makedirs(os.path.dirname(filename), exist_ok=True)
#
#     # Save as float32 (C's float) for most tensors
#     if tensor.dtype in [torch.float32, torch.float]:
#         tensor.detach().cpu().numpy().astype(np.float32).tofile(filename)
#     # Save as int32 for integer tensors
#     elif tensor.dtype in [torch.int32, torch.int64, torch.long, torch.int]:
#         tensor.detach().cpu().numpy().astype(np.int32).tofile(filename)
#     else:
#         print(f"Warning: Unhandled dtype {tensor.dtype} for {filename}")
#         tensor.detach().cpu().numpy().tofile(filename)
#
#     # Also save shape information for easier loading in C
#     with open(f"{filename}.shape", "w") as f:
#         f.write(",".join(str(dim) for dim in tensor.shape))


db_path = "bins/weights/encoder_layer_0_attention_output_dense_bias.bin"
dw_path = "bins/weights/encoder_layer_0_attention_output_dense_weight.bin"
lnw_path = "bins/weights/encoder_layer_0_attention_output_LayerNorm_weight.bin"
lnb_path = "bins/weights/encoder_layer_0_attention_output_LayerNorm_bias.bin"


class BertSelfOutput(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.dense.weight.data = torch.from_numpy(np.fromfile(
            dw_path, dtype=np.float32)).view(config.hidden_size, config.hidden_size)
        self.dense.bias.data = torch.from_numpy(
            np.fromfile(db_path, dtype=np.float32))
        self.LayerNorm.weight.data = torch.from_numpy(
            np.fromfile(lnw_path, dtype=np.float32))
        self.LayerNorm.bias.data = torch.from_numpy(
            np.fromfile(lnb_path, dtype=np.float32))

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        breakpoint()
        # Save inputs
        # save_tensor_as_bin(
        #     f"bins/layer{self.layer_idx}_self_output_hidden_states.bin", hidden_states)
        # save_tensor_as_bin(
        #     f"bins/layer{self.layer_idx}_self_output_input_tensor.bin", input_tensor)

        # Linear projection
        hidden_states = self.dense(hidden_states)
        # save_tensor_as_bin(
        #     f"bins/layer{self.layer_idx}_self_output_dense.bin", hidden_states)

        # Dropout (in eval mode, this is identity)
        if not self.training:
            hidden_states = self.dropout(hidden_states)
            # save_tensor_as_bin(
            #     f"bins/layer{self.layer_idx}_self_output_dropout.bin", hidden_states)

        # Residual connection
        residual = hidden_states + input_tensor
        # save_tensor_as_bin(
        #     f"bins/layer{self.layer_idx}_self_output_residual.bin", residual)

        # Layer normalization
        normalized = self.LayerNorm(residual)
        # save_tensor_as_bin(
        #     f"bins/layer{self.layer_idx}_self_output_layernorm.bin", normalized)

        return normalized


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

    # save_tensor_as_bin(
    #     f"bins/layer{self.layer_idx}_self_output_hidden_states.bin", hidden_states)
    # save_tensor_as_bin(
    #     f"bins/layer{self.layer_idx}_self_output_input_tensor.bin", input_tensor)


if __name__ == "__main__":
    B, T, C = 2, 128, 768
    config = BertConfig()
    h_path = "bins/t1.bin"
    x_path = "bins/tw.bin"
    x = torch.from_numpy(np.fromfile(x_path, dtype=np.float32)).view(B, T, C)
    h = torch.from_numpy(np.fromfile(h_path, dtype=np.float32)).view(B, T, C)
    breakpoint()

    print(f"shape of x: {x.shape}")
    so = BertSelfOutput(config)
    out = so(h, x)
    print(out)
