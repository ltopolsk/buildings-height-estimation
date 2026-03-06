import torch

input_ckpt = 'solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth'
output_ckpt = 'solov2_feature_fusion_init.pth'

print(f"[*] Copying weights: {input_ckpt}")
checkpoint = torch.load(input_ckpt, map_location='cpu')
state_dict = checkpoint['state_dict']

new_state_dict = {}

for key, value in state_dict.items():
    if key.startswith('backbone.'):
        rgb_key = key.replace('backbone.', 'backbone.rgb_backbone.')
        new_state_dict[rgb_key] = value.clone()
        
        sar_key = key.replace('backbone.', 'backbone.sar_backbone.')
        if 'conv1.weight' in key:
            sar_value = value.mean(dim=1, keepdim=True)
            new_state_dict[sar_key] = sar_value
        else:
            new_state_dict[sar_key] = value.clone()
    else:
        new_state_dict[key] = value

checkpoint['state_dict'] = new_state_dict
torch.save(checkpoint, output_ckpt)
print(f"[+] Copying done, new weights are in: {output_ckpt}")
