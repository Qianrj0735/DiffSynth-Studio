#!/usr/bin/env python3
import os
import argparse

def main(train_dir: str, output_csv: str, text: str):
    # 确保 train 目录存在
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    # 列出所有文件（非隐藏文件），并按名字排序
    files = sorted(f for f in os.listdir(train_dir) if not f.startswith('.'))
    if not files:
        print(f"No files found in {train_dir}. Exiting.")
        return

    # 先对 text 中的双引号做转义（CSV 里双引号需要写成 ""）
    escaped_text = text.replace('"', '""')

    # 写入 metadata.csv
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        # 写表头
        f.write('file_name,text\n')
        # 写每一行：file_name,"escaped_text"
        for fname in files:
            f.write(f'{fname},"{escaped_text}"\n')

    print(f"Generated metadata at {output_csv}, total {len(files)} entries.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate metadata.csv with the text field always in double quotes.'
    )
    parser.add_argument(
        '--train_dir', '-i',
        default='data/example_dataset/train',
        help='Path to the train/ directory containing .mp4/.jpg files'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/example_dataset/metadata.csv',
        help='Path to write metadata.csv'
    )
    parser.add_argument(
        '--text', '-t',
        required=True,
        help='The description text to associate with every file'
    )
    args = parser.parse_args()
    main(args.train_dir, args.output, args.text)
