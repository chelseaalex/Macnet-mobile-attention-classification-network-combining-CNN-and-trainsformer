import torch
from swin_transformer_pytorch import SwinTransformer

ndet = SwinTransformer(
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heas=(3, 6, 12, 24),
    channels=3,
    num_classes=4,
    head_dim=32,
    window_size=8,
    downscaling_factors=(4, 2, 2, 2),
    relative_pos_embedding=True
)
dummy_x = torch.randn(1, 3, 256, 256)
logits = net(dummy_x)  # (1,3)
print(net)
print(logits)
