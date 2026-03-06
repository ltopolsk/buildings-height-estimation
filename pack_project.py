import os
import zipfile

def pack_project(output_filename='solov2_mgr_project.zip'):
    print("[*] Packing project ...")
    
    ignore_dirs = {'__pycache__', '.git', 'work_dirs', 'runs', 'dataset', '.ipynb_checkpoints', '.vscode', ''}
    
    keep_weights = {
        'solov2_r50_fpn_4channel_init.pth',
        'solov2_feature_fusion_init.pth',
        'solov2_sar_only_init.pth',
    }

    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.endswith('.pth') and file not in keep_weights:
                    continue
                    
                if file.endswith('.zip'):
                    continue
                    
                arcname = os.path.relpath(file_path, '.')
                zipf.write(file_path, arcname)
                
    print(f"[+] Project packet into: {output_filename}")

if __name__ == '__main__':
    pack_project()