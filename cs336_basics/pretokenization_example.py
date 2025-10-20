import os
from typing import BinaryIO
import regex
from multiprocessing import Pool
from collections import Counter,defaultdict

PAT = ""
compile_pat = regex.compile(PAT)


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

def process_chunk(
    task:tuple
) -> dict:
    input_path,start,end = task
    local_word_freqs = defaultdict(int)


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

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.






