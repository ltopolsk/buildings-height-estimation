import torch

input_ckpt = 'solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth'
output_ckpt = 'solov2_r50_fpn_4channel_init.pth'

print(f"[*] Ładowanie oryginalnych wag z: {input_ckpt}")
checkpoint = torch.load(input_ckpt, map_location='cpu')
state_dict = checkpoint['state_dict']

target_layer = 'backbone.conv1.weight'
conv1_weight = state_dict[target_layer] # [64, 3, 7, 7]
print(f"[*] Oryginalny kształt warstwy {target_layer}: {conv1_weight.shape}")

out_channels, in_channels, h, w = conv1_weight.shape
new_conv1_weight = torch.zeros((out_channels, 4, h, w), dtype=conv1_weight.dtype)

new_conv1_weight[:, :3, :, :] = conv1_weight
new_conv1_weight[:, 3:4, :, :] = conv1_weight.mean(dim=1, keepdim=True)

state_dict[target_layer] = new_conv1_weight
print(f"[*] Nowy kształt warstwy {target_layer}: {state_dict[target_layer].shape}")

torch.save(checkpoint, output_ckpt)
print(f"[+] Zapisano zoperowane wagi do: {output_ckpt}")