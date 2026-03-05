import torch

input_ckpt = 'solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth'
output_ckpt = 'solov2_sar_only_init.pth'

print(f"[*] Klonowanie wag dla eksperta SAR...")
checkpoint = torch.load(input_ckpt, map_location='cpu')
state_dict = checkpoint['state_dict']

target_layer = 'backbone.conv1.weight'
conv1_weight = state_dict[target_layer]

# Kompresja wiedzy z 3 kanałów RGB do 1 kanału
new_conv1 = conv1_weight.mean(dim=1, keepdim=True)
state_dict[target_layer] = new_conv1

torch.save(checkpoint, output_ckpt)
print(f"[+] Gotowe: {output_ckpt}")
