import os
import numpy as np
from tqdm import tqdm

from Modules.bpe_tokenizer import BPETokenizer
from cs336_basics.pretokenization_example import special_tokens
from tests.adapters import run_train_bpe

DATA_DIR = 'data'
INPUT_FILES = {
    'train': 'TinyStoriesV2-GPT4-train.txt',
    'val': 'TinyStoriesV2-GPT4-valid.txt'
}
output_dir = os.path.join(DATA_DIR, "bin")
os.makedirs(output_dir, exist_ok=True)

def process(tokenizer):
    def file_reader(path, read_buffer_size=64 * 1024):
        with open(path, 'r', encoding='utf-8') as f:
            while True:
                data = f.read(read_buffer_size)
                if not data:
                    break
                yield data
    for split,filename in INPUT_FILES.items():
        file_path = os.path.join(DATA_DIR,filename)
        if not os.path.exists(file_path):
            print(f"跳过: 找不到文件 {file_path}")
            continue
        print(f"正在处理 {split} 数据: {filename} ...")

        os.makedirs(output_dir, exist_ok=True)
        bin_path = os.path.join(output_dir, f'{split}.bin')
        with open(bin_path, 'wb') as f_out:

            buffer = []  # 内存缓冲区
            token_count = 0

            # 4. 调用你的 encode_iterable
            # 这里把 file_reader 生成器传进去，实现真正的流式
            # tqdm 用于显示进度 (total 只能估算，或者设为 None 仅仅显示速率)
            iterator = tokenizer.encode_iterable(file_reader(file_path))

            for token_id in tqdm(iterator, desc="Tokenizing"):
                buffer.append(token_id)

                # 5. 缓冲区满了就写入磁盘
                if len(buffer) >= 1024*1024:
                    arr = np.array(buffer, dtype=np.uint16)
                    arr.tofile(f_out)  # 追加写入到文件末尾
                    token_count += len(buffer)
                    buffer = []  # 清空缓冲区，释放内存

            if buffer:
                arr = np.array(buffer, dtype=np.uint16)
                arr.tofile(f_out)
                token_count += len(buffer)
            print(f"\nDone! Total tokens: {token_count}")
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    vocab, merges = run_train_bpe(os.path.join(DATA_DIR, INPUT_FILES['train']), 10000, special_tokens)
    tokenizer = BPETokenizer(vocab,merges,special_tokens)

    process(tokenizer)

if __name__ == '__main__':
    main()