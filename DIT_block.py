import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention,Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DITBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super(DITBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_size, act_layer=approx_gelu)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa)) * gate_msa.unsqueeze(1)
        x = x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)) * gate_mlp.unsqueeze(1)
        return x

    
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super(FinalLayer, self).__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x
    
if __name__ == "__main__":
    hidden_size = 256
    num_heads = 8
    mlp_ratio = 4.0
    patch_size = 2
    out_channels = 3
    seq_len = 64   # 假设有 64 个 patch
    batch_size = 4

    # 输入 token 向量 [B, N, C]
    x = torch.randn(batch_size, seq_len, hidden_size)
    # 条件嵌入向量 [B, C]，用于调制
    c = torch.randn(batch_size, hidden_size)

    # 测试 DITBlock
    dit_block = DITBlock(hidden_size, num_heads, mlp_ratio)
    out_block = dit_block(x, c)
    print("DITBlock 输出 shape:", out_block.shape)  # [B, N, C]

    # 测试 FinalLayer
    final_layer = FinalLayer(hidden_size, patch_size, out_channels)
    out_final = final_layer(out_block, c)
    print("FinalLayer 输出 shape:", out_final.shape)  # [B, N, patch_size * patch_size * out_channels]

