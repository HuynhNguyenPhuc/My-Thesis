import os
import hashlib
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd

DRIVE_FOLDER = 'https://drive.google.com/drive/folders/1fb5gGBxFIibvHrsJGquO6N8rqKSbkIZB'
RAW_SUBDIR = os.path.join('raw', '3dmodels', 'original')


def add_args(parser: argparse.ArgumentParser):
    pass


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def get_metadata(output_dir, **kwargs):
    """
    Index the FBX files under <output_dir>/raw/3dmodels/original/BATCH_*/Set_*/.

    Houses3K is distributed through Google Drive (see DRIVE_FOLDER), so there is
    no metadata file to fetch: the files are scanned and hashed instead.
    """
    raw_dir = os.path.join(output_dir, RAW_SUBDIR)
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f'{raw_dir} not found. Download Houses3K first (see docs/House3K.md): {DRIVE_FOLDER}')

    files = []
    for root, _, names in os.walk(raw_dir):
        for name in names:
            if name.lower().endswith('.fbx'):
                files.append(os.path.join(root, name))
    files.sort()

    with ThreadPoolExecutor() as executor:
        hashes = list(tqdm(executor.map(_sha256, files), total=len(files), desc='Hashing files'))

    records = []
    for path, sha256 in zip(files, hashes):
        records.append({
            'sha256': sha256,
            'file_identifier': os.path.relpath(path, raw_dir).replace(os.sep, '/'),
            'local_path': os.path.relpath(path, output_dir).replace(os.sep, '/'),
        })
    return pd.DataFrame.from_records(records).drop_duplicates('sha256')


def download(metadata, output_dir, **kwargs):
    """Files are already local; just report their paths so build_metadata can merge them."""
    missing = [p for p in metadata['local_path'] if not os.path.exists(os.path.join(output_dir, p))]
    if missing:
        raise FileNotFoundError(f'{len(missing)} files are missing, e.g. {missing[0]}')
    return metadata[['sha256', 'local_path']]


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    metadata = metadata.to_dict('records')
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
                tqdm(total=len(metadata), desc=desc) as pbar:
            def worker(metadatum):
                try:
                    record = func(os.path.join(output_dir, metadatum['local_path']), metadatum['sha256'])
                    if record is not None:
                        records.append(record)
                except Exception as e:
                    print(f'Error processing object {metadatum["sha256"]}: {e}')
                pbar.update()

            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except Exception:
        print('Error happened during processing.')
    return pd.DataFrame.from_records(records)
