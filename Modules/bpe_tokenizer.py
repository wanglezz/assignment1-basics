import os
from typing import (
    Iterable,
    Iterator,
    List,
    Dict,
    Optional,
    Tuple,
    Union,
    Self  # Python 3.11+ (或从 typing_extensions 导入)
)
import json
import regex
import re

from submitit.test_pickle import hello_fn


class BPETokenizer:
    """
    一个基于 BPE 算法的标记器（Tokenizer）类。

    Attributes:
        vocab (Dict[int, bytes]): 词汇表，从 ID 映射到 bytes。
        merges (List[Tuple[bytes, bytes]]): 合并规则。
        special_tokens (List[str]): 特殊标记及其对应的 ID。
    """

    def __init__(
            self,
            vocab: Dict[int, bytes],
            merges: List[Tuple[bytes, bytes]],  # 假设 merges 是 (b1, b2) -> rank 的字典
            special_tokens: Optional[list[str]] = None
    ):
        """
        初始化 BPE Tokenizer。

        Args:
            vocab: 从 token ID 到 byte 形式的 token 的映射字典。
            merges: BPE 合并规则字典，(byte1, byte2) -> 优先级。
            special_tokens: 可选的特殊标记字典, str -> ID。
        """
        self.decoder_vocab = vocab
        self.encoder_vocab: Dict[bytes, int] = {b: i for i, b in vocab.items()}
        self.merges_ranks: Dict[Tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }
        self.special_tokens = special_tokens if special_tokens else {}
        self.special_tokens_set = set(self.special_tokens)

        # cache bytes->token_id_seq
        self.bpe_cache:Dict[bytes,List[int]] = {}

        gpt2_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # GPT2 Regex
        self.compile_pat = regex.compile(gpt2_pat)

        if self.special_tokens:
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            special_pattern_str = "|".join(re.escape(s) for s in sorted_special_tokens)
            self.split_pattern = re.compile(f'({special_pattern_str})')
        else:
            self.split_pattern = None



    @classmethod
    def from_files(
            cls,
            vocab_filepath: Union[str, os.PathLike],
            merges_filepath: Union[str, os.PathLike],
            special_tokens: Optional[list[str]] = None
    ) -> Self:  # 返回类型使用 Self (或 "BPETokenizer")
        """
        从词汇表和合并规则文件加载 Tokenizer。

        Args:
            vocab_filepath: 词汇表文件路径 (例如 vocab.json)。
            merges_filepath: 合并规则文件路径 (例如 merges.txt)。
            special_tokens: 可选的特殊标记字典。

        Returns:
            一个 BPETokenizer 的新实例。
        """
        # ... 此处是加载文件并解析为 vocab 和 merges 的逻辑 ...
        # ... 例如:
        # vocab, merges = cls._load_files(vocab_filepath, merges_filepath)

        # 使用 cls(...) 来调用 __init__ 并创建实例
        # return cls(vocab, merges, special_tokens)
        vocab_encode = {} # Dict{str,int}
        merges = []
        try:
            with open(vocab_filepath,"r", encoding="utf-8") as f:
                vocab_encode = json.load(f)
            if not isinstance(vocab_encode, dict):
                print(f"警告: {vocab_filepath} 加载的内容不是一个字典 (dict)，而是 {type(vocab_encode)}")
        except FileNotFoundError:
            print(f"错误: 找不到文件 {vocab_filepath}")
        except json.JSONDecodeError:
            print(f"错误: {vocab_filepath} 不是一个有效的 JSON 文件。")
        except Exception as e:
            print(f"加载 {vocab_filepath} 时发生未知错误: {e}")

        try:
            with open(merges_filepath,"r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    parts = line.split(' ')
                    if len(parts) == 2:
                        part1_bytes = parts[0].encode('utf-8')
                        part2_bytes = parts[1].encode('utf-8')
                        merges.append((part1_bytes, part2_bytes))
                    else:
                        print(f"警告: 忽略格式错误的行 (未找到恰好2个部分): {line!r}")
        except FileNotFoundError:
            print(f"错误: 找不到文件 {merges_filepath}")
        except Exception as e:
            print(f"加载 {merges_filepath} 时发生未知错误: {e}")

        # 将 str 转为 bytes
        vocab = {id:token.encode("utf-8") for token,id in vocab_encode.items()}

        return cls(vocab,merges,special_tokens)

    def merge(
            self,
            word_bytes: bytes,
    )->List[int]:
        token_bytes_seq = [bytes([b]) for b in word_bytes]
        if not token_bytes_seq:
            return []
        while True:
            pairs = list(zip(token_bytes_seq[:-1], token_bytes_seq[1:]))
            best_rank = float("inf")
            best_pair = None
            for pair in pairs:
                if pair in self.merges_ranks:
                    rank = self.merges_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
            if best_pair is None:
                break
            new_seq = []
            a, b = best_pair
            i = 0
            bytes_len = len(token_bytes_seq)
            while i < bytes_len:
                if i < bytes_len - 1 and token_bytes_seq[i] == a and token_bytes_seq[i + 1] == b:
                    new_seq.append(a + b)
                    i += 2
                else:
                    new_seq.append(token_bytes_seq[i])
                    i += 1
            token_bytes_seq = new_seq
        token_id_seq = []
        for token_bytes in token_bytes_seq:
            if token_bytes in self.encoder_vocab:
                token_id_seq.append(self.encoder_vocab[token_bytes])
            else:
                print(f"警告: 字节 {token_bytes} (来自 {word_bytes!r}) 不在词汇表中")
        self.bpe_cache[word_bytes] = token_id_seq
        return token_id_seq

    def encode(self, text: str) -> List[int]:
        """
        将一个字符串编码为 token ID 列表。

        Args:
            text: 输入的原始字符串。

        Returns:
            一个由 token ID 组成的列表。
        """
        # ... 实现 Pre-tokenization 和 BPE 编码逻辑 ...
        final_ids: List[int] = []

        # --- Pre-tokenization ---
        words: List[str] = []
        if self.split_pattern :
            text_parts = self.split_pattern.split(text)
        else :
            text_parts = [text]

        for part in text_parts:
            if not part:
                continue
            if part in self.special_tokens_set:
                words.append(part)
            else:
                for match in self.compile_pat.finditer(part):
                    words.append(match.group(0))

        # --- BPE encode ---
        for word in words:
            if word in self.special_tokens_set:
                final_ids.append(self.encoder_vocab[word.encode('utf-8')])
                continue
            word_bytes = word.encode('utf-8')
            if word_bytes in self.bpe_cache:
                final_ids.extend(self.bpe_cache[word_bytes])
                continue

            # --- merge ---
            token_id_seq = self.merge(word_bytes)
            self.bpe_cache[word_bytes] = token_id_seq
            final_ids.extend(token_id_seq)
        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        以节省内存的方式，懒加载地对一个字符串迭代器进行编码。

        Args:
            iterable: 一个字符串的可迭代对象 (例如一个文件句柄)。

        Yields:
            逐个产出 token ID (int)。
        """
        remainder = ""
        for chunk in iterable:
            text_to_process = remainder + chunk
            words_to_process = []
            if self.split_pattern:
                text_parts = self.split_pattern.split(text_to_process)
            else:
                text_parts = [text_to_process]  # 没有 special tokens, 视为一个 part
            for part in text_parts[:-1]:
                if part in self.special_tokens_set:
                    words_to_process.append(part)
                else:
                    matches = list(self.compile_pat.finditer(part))
                    for match in matches:
                        words_to_process.append(match.group(0))
            remainder = text_parts[-1]
            for word in words_to_process:
                if word in self.special_tokens_set:
                    yield self.encoder_vocab[word.encode('utf-8')]
                    continue

                word_bytes = word.encode('utf-8')

                if word_bytes in self.bpe_cache:
                    for token_id in self.bpe_cache[word_bytes]:
                        yield token_id
                    continue

                # --- merge ---
                token_id_seq = self.merge(word_bytes)
                for token_id in token_id_seq:
                    yield token_id
                self.bpe_cache[word_bytes] = token_id_seq
        if remainder:
            final_ids = self.encode(remainder)
            for token_id in final_ids:
                yield token_id
    def decode(self, ids: List[int]) -> str:
        """
        将一个 token ID 列表解码回原始字符串。

        Args:
            ids: 一个 token ID 列表。

        Returns:
            解码后的字符串。
        """
        # ... 实现解码逻辑 (ID -> bytes -> str) ...
        bytes_chunks = []
        for token_id in ids:
            if token_id in self.decoder_vocab:
                bytes_chunks.append(self.decoder_vocab[token_id])
            else:
                pass
        full_bytes = b"".join(bytes_chunks)
        try:
            text = full_bytes.decode('utf-8')
            return text
        except Exception as e:
            print(f"解码时发生严重错误: {e}")
            return ""  # 或者返回一个错误标记