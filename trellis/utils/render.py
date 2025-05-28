import os
import json
import numpy as np
from subprocess import DEVNULL, call
from concurrent.futures import ProcessPoolExecutor, as_completed

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def render(file_path, output_dir, num_views=4):
    shape_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = os.path.join(output_dir, shape_name)
    os.makedirs(output_folder, exist_ok=True)

    views = [
        {'yaw': 0.0, 'pitch': 0.0, 'radius': 2, 'fov': 40 / 180 * np.pi},    # Front
        {'yaw': np.pi, 'pitch': 0.0, 'radius': 2, 'fov': 40 / 180 * np.pi},  # Back
        {'yaw': np.pi / 2, 'pitch': 0.0, 'radius': 2, 'fov': 40 / 180 * np.pi}, # Left
        {'yaw': -np.pi / 2, 'pitch': 0.0, 'radius': 2, 'fov': 40 / 180 * np.pi} # Right
    ]

    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), '../../dataset_toolkits/blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
    ]


    call(args, stdout=DEVNULL, stderr=DEVNULL)
    print(f'Finished rendering {file_path}')

def render_all_shapes(shape_dir, output_dir, num_views=4, num_workers=4):
    os.makedirs(output_dir, exist_ok=True)

    print('Checking Blender installation...', flush=True)
    _install_blender()

    glb_files = [f for f in os.listdir(shape_dir) if f.lower().endswith('.glb')]
    print(f'Found {len(glb_files)} .glb files in {shape_dir}')

    file_paths = [os.path.join(shape_dir, f) for f in glb_files]

    max_workers = min(num_workers, len(file_paths))
    print(f'Starting rendering with {max_workers} parallel workers...', flush=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(render, file_path, output_dir, num_views) 
                   for file_path in file_paths]

        for future in as_completed(futures):
            future.result()  

    print('All renderings completed.')
