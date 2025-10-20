import os
import re
from typing import BinaryIO
import regex
from multiprocessing import Pool
from collections import Counter,defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" #GPT2 Regex
compile_pat = regex.compile(PAT)
special_tokens = ["<|endoftext|>"]
special_pattern_str = "|".join(re.escape(s) for s in special_tokens)
split_pattern = re.compile(f'({special_pattern_str})')


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# pre-tokenize chunk
def process_chunk(
    task:tuple
) -> dict:
    input_path,start,end = task
    try:
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_str = chunk_bytes.decode("utf-8",errors="ignore")
    except Exception as e:
        print(f"Reading Chunk {start}-{end}: {e}")
        return defaultdict(int)

    local_word_freqs = defaultdict(int)

    if split_pattern and special_pattern_str:
        text_parts = split_pattern.split(chunk_str)
    else:
        text_parts = [chunk_str]

    for part in text_parts:
        if not part:
            continue

        if part in special_tokens:
            local_word_freqs[part] += 1
        else:
            for match in compile_pat.finditer(part):
                local_word_freqs[match.group(0)] += 1

    return local_word_freqs


def train_bpe(input_path:str,
    vocab_size:int,
    special_token:list[str],
)->(dict,list):
    # initialize vocab with special_tokens and basic 256 byte
    vocab = {i:bytes([i])for i in range(256)}
    current_id = 256
    for token in special_token:
        token_bytes = token.encode("utf-8")
        vocab[current_id] = token_bytes
        current_id += 1
    merges = []
    # --- pre-tokenization stage ---

    # divide raw_text into chunk
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start < end:

            tasks.append((input_path, start, end))

    global_word_freqs = Counter()

    with Pool(processes=num_processes) as pool:
        for local_word_freqs in pool.imap_unordered(process_chunk, tasks):
            global_word_freqs.update(local_word_freqs)

    stats = {} #  dict[tuple[bytes],int]
    for word_str, freq in global_word_freqs.items():
        token_id_sequence = tuple(word_str.encode("utf-8"))
        stats[token_id_sequence] = freq

    # --- bpe merge stage ---


if __name__ == '__main__':

    input_path = os.path.join("..","data", "TinyStoriesV2-GPT4-valid.txt")

    train_bpe(input_path,
    4096,
    special_tokens)







