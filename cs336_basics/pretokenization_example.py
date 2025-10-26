import os
import time
import json
import cProfile
import pstats
from tests.adapters import run_train_bpe

special_tokens = ["<|endoftext|>"]
def save(
        vocab,
        merges
):
    vocab_for_json = {
        token_bytes.decode('latin-1'): token_id
        for token_id, token_bytes in vocab.items()
    }

    # 2. 将这个新字典保存为 JSON
    with open("TinyStories/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_for_json, f, ensure_ascii=False, indent=2)

    print(f"成功将 {len(vocab_for_json)} 个条目保存到 vocab.json")

    with open("TinyStories/merges.txt", "w", encoding="utf-8") as f:
        # 写入头部
        f.write("# bpe-merges v1.0\n")

        # 遍历 merges 列表
        for (token1_bytes, token2_bytes) in merges:
            # 总是解码两个 token
            token1_str = token1_bytes.decode('latin-1')
            token2_str = token2_bytes.decode('latin-1')

            # 总是用空格将它们隔开，并写入一行
            f.write(f"{token1_str} {token2_str}\n")

    print(f"成功将 {len(merges)} 条合并规则保存到 merges.txt")

if __name__ == '__main__':

    # input_path = os.path.join("..","tests","fixtures", "corpus.en")
    # input_path = os.path.join("..","data","TinyStoriesV2-GPT4-train.txt")
    input_path = os.path.join("..","data","owt_train.txt")

    pr = cProfile.Profile()
    pr.enable()  # 开始记录
    start_time = time.time()
    vocab,merges = run_train_bpe(input_path,
    32000,
    special_tokens)
    save(vocab,merges)
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time}")
    pr.disable()
    # 打印结果（通常使用 pstats 模块格式化输出）
    stats = pstats.Stats(pr).sort_stats('cumulative')  # 按累积时间排序
    stats.print_stats(10)  # 打印前 10 行







