import os
import shutil

source_root = './outputs/gaussiandreamer-sd'
target_folder = './gifs'
os.makedirs(target_folder, exist_ok=True)

# Walk through all folders
for root, dirs, files in os.walk(source_root):
    for file in files:
        if file == 'shape.gif':
            full_path = os.path.join(root, file)

            # Confirm the parent folder is named 'save'
            if os.path.basename(os.path.dirname(full_path)) == 'save':
                parent_dir = os.path.dirname(os.path.dirname(full_path))  # Get the dir_n (just above 'save')

                # Relative path from source_root to the directory (dir1/dir2/..../dir_n)
                rel_path = os.path.relpath(parent_dir, source_root)

                # Replace / with ' or ' for the filename
                safe_name = rel_path.replace(os.sep, '_or_')
                target_path = os.path.join(target_folder, f'{safe_name}.gif')

                shutil.copyfile(full_path, target_path)
                print(f'Copied: {full_path} → {target_path}')
