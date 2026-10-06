import os
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge captioned_<rank>.csv files (from extract_caption.py) into metadata.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing metadata.csv and captioned_*.csv')
    opt = parser.parse_args()

    metadata_path = os.path.join(opt.output_dir, 'metadata.csv')
    metadata = pd.read_csv(metadata_path).set_index('sha256', drop=False)
    if 'captions' not in metadata.columns:
        metadata['captions'] = None
    metadata['captions'] = metadata['captions'].astype(object)

    files = [f for f in os.listdir(opt.output_dir) if f.startswith('captioned_') and f.endswith('.csv')]
    merged = 0
    for f in sorted(files):
        df = pd.read_csv(os.path.join(opt.output_dir, f)).dropna(subset=['captions'])
        for sha256, captions in zip(df['sha256'], df['captions']):
            if sha256 in metadata.index:
                metadata.at[sha256, 'captions'] = captions
                merged += 1
    metadata.to_csv(metadata_path, index=False)
    print(f'Merged captions for {merged} objects from {len(files)} file(s)')
