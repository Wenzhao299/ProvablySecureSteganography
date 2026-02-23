import argparse
import csv
import json
import os
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from config import (
    DEFAULT_AC_PRECISION,
    DEFAULT_DISCOP_BASELINE,
    DEFAULT_IMEC_BLOCK_SIZE,
    DEFAULT_INPUT_CSV,
    DEFAULT_MAX_CONTEXTS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MESSAGE_BITS_LENGTH,
    DEFAULT_METHOD,
    DEFAULT_METEOR_REORDER,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PRECISION,
    DEFAULT_SEED,
    DEFAULT_SPARSAMP_BLOCK_SIZE,
    DEFAULT_TEMP,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    SUPPORTED_METHODS,
    SUPPORTED_MODEL_NAMES,
    get_parallel_paths,
)
from utils import (
    decode_with_method,
    encode_context,
    encode_with_method,
    initialize_model_and_precision,
    random_bits,
)


def split_contexts_by_rank(contexts: List[Dict[str, str]], rank: int, world_size: int) -> List[Dict[str, str]]:
    total = len(contexts)
    start = (total * rank) // world_size
    end = (total * (rank + 1)) // world_size
    return contexts[start:end]


def worker(rank: int, world_size: int, args, contexts: List[Dict[str, str]]):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    local_seed = args.seed + rank
    random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)

    enc, model = initialize_model_and_precision(args, seed=local_seed)
    model.to(device)
    model.eval()

    part_jsonl_path = os.path.join(args.tmp_dir, f"{args.method}.part_{rank}.jsonl")
    part_stats_path = os.path.join(args.tmp_dir, f"{args.method}.stats.{args.run_tag}.{rank}.json")
    worker_contexts = split_contexts_by_rank(contexts, rank, world_size)

    matched = 0
    processed = 0
    failed = 0

    with open(part_jsonl_path, "a", encoding="utf-8") as f_out:
        for item in tqdm(worker_contexts, desc=f"GPU {rank}", position=rank, unit="sample"):
            context_id = item["id"]
            context_text = item["text"]

            try:
                context_tokens = encode_context(context_text, enc)
                if not context_tokens:
                    raise ValueError("Context tokenization is empty.")

                message_bits = random_bits(args.message_bits_length)
                generated_ids, embedded_bits = encode_with_method(
                    args.method,
                    model,
                    context_tokens,
                    message_bits,
                    args,
                    device,
                )

                stego_text = enc.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                extracted_bits = decode_with_method(
                    args.method,
                    model,
                    context_tokens,
                    generated_ids,
                    args,
                    device,
                )

                is_match = extracted_bits.startswith(embedded_bits) if embedded_bits else True
                if is_match:
                    matched += 1

                jsonl_record = {
                    "id": context_id,
                    "text": stego_text,
                }
                f_out.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")
                processed += 1
                if args.flush_every > 0 and (processed % args.flush_every == 0):
                    f_out.flush()

            except Exception as exc:
                failed += 1
                tqdm.write(f"[GPU {rank}] context_id={context_id} failed: {exc}")

    with open(part_stats_path, "w", encoding="utf-8") as f_stats:
        json.dump(
            {
                "rank": rank,
                "processed": processed,
                "matched": matched,
                "failed": failed,
            },
            f_stats,
            ensure_ascii=False,
        )


def load_contexts(input_csv: str, max_contexts: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(input_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            rows.append({"id": str(row[0]), "text": row[1]})

    if max_contexts is not None and max_contexts > 0:
        rows = rows[:max_contexts]
    return rows


def _parse_jsonl_id_text(line: str) -> Optional[Tuple[str, str]]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None
    if "id" not in obj or "text" not in obj:
        return None

    row_id = str(obj["id"]).strip()
    if not row_id:
        return None

    return row_id, str(obj["text"])


def list_part_jsonl_paths(args) -> List[str]:
    prefix = f"{args.method}.part_"
    part_paths: List[str] = []
    if not os.path.isdir(args.tmp_dir):
        return part_paths

    for file_name in sorted(os.listdir(args.tmp_dir)):
        if file_name.startswith(prefix) and file_name.endswith(".jsonl"):
            part_paths.append(os.path.join(args.tmp_dir, file_name))
    return part_paths


def collect_processed_ids(args, output_jsonl_path: str) -> Set[str]:
    processed_ids: Set[str] = set()

    def _collect_ids_from_jsonl(path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = _parse_jsonl_id_text(line)
                if parsed is None:
                    continue
                row_id, _ = parsed
                processed_ids.add(row_id)

    _collect_ids_from_jsonl(output_jsonl_path)
    for part_jsonl_path in list_part_jsonl_paths(args):
        _collect_ids_from_jsonl(part_jsonl_path)

    return processed_ids


def merge_part_files(args, run_tag: Optional[str], world_size: int, output_jsonl_path: str) -> Tuple[int, int, int, int]:
    merged_rows: List[Tuple[str, str]] = []
    seen_ids: Set[str] = set()

    def _append_rows_from_jsonl(path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                parsed = _parse_jsonl_id_text(line)
                if parsed is None:
                    continue
                row_id, text = parsed
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                merged_rows.append((row_id, text))

    _append_rows_from_jsonl(output_jsonl_path)

    part_jsonl_paths = list_part_jsonl_paths(args)
    for part_jsonl_path in part_jsonl_paths:
        _append_rows_from_jsonl(part_jsonl_path)

    with open(output_jsonl_path, "w", encoding="utf-8") as f_out:
        for row_id, text in merged_rows:
            f_out.write(json.dumps({"id": row_id, "text": text}, ensure_ascii=False) + "\n")

    for part_jsonl_path in part_jsonl_paths:
        os.remove(part_jsonl_path)

    total_processed = 0
    total_matched = 0
    total_failed = 0

    if run_tag is not None:
        for rank in range(world_size):
            part_stats_path = os.path.join(args.tmp_dir, f"{args.method}.stats.{run_tag}.{rank}.json")
            if os.path.exists(part_stats_path):
                with open(part_stats_path, "r", encoding="utf-8") as f_stats:
                    stats = json.load(f_stats)
                total_processed += int(stats.get("processed", 0))
                total_matched += int(stats.get("matched", 0))
                total_failed += int(stats.get("failed", 0))
                os.remove(part_stats_path)

    return total_processed, total_matched, total_failed, len(merged_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch steganography + extraction with multi-GPU parallelism")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=SUPPORTED_METHODS)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, choices=SUPPORTED_MODEL_NAMES)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--max_contexts", type=int, default=DEFAULT_MAX_CONTEXTS)
    parser.add_argument("--message_bits_length", type=int, default=DEFAULT_MESSAGE_BITS_LENGTH)
    parser.add_argument("--flush_every", type=int, default=1, help="Flush part JSONL every N successful samples.")

    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--model_precision", type=str, default=DEFAULT_MODEL_PRECISION)
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    parser.add_argument("--ac_precision", type=int, default=DEFAULT_AC_PRECISION)
    parser.add_argument("--discop_baseline", action="store_true", default=DEFAULT_DISCOP_BASELINE)
    parser.add_argument("--imec_block_size", type=int, default=DEFAULT_IMEC_BLOCK_SIZE)
    parser.add_argument("--meteor_reorder", action="store_true", default=DEFAULT_METEOR_REORDER)
    parser.add_argument("--sparsamp_block_size", type=int, default=DEFAULT_SPARSAMP_BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("stego_parallel.py requires CUDA GPUs.")

    if args.message_bits_length <= 0:
        args.message_bits_length = args.max_tokens * 16

    _, args.output_dir, args.tmp_dir, output_jsonl_path = get_parallel_paths(args.input_csv, args.method)
    os.makedirs(args.tmp_dir, exist_ok=True)

    contexts = load_contexts(args.input_csv, args.max_contexts)
    if not contexts:
        raise ValueError(f"No valid data found in {args.input_csv}")

    processed_ids = collect_processed_ids(args, output_jsonl_path)
    contexts_to_process = [item for item in contexts if item["id"] not in processed_ids]

    if processed_ids:
        print(
            f"Resume mode: found {len(processed_ids)} processed ids "
            f"(from existing output/temp shards)."
        )
    print(f"Remaining contexts to process: {len(contexts_to_process)} / {len(contexts)}")

    world_size = torch.cuda.device_count()
    args.run_tag = str(int(time.time()))
    if contexts_to_process:
        print(f"Launching {world_size} worker(s) for {len(contexts_to_process)} contexts...")
        mp.spawn(worker, nprocs=world_size, args=(world_size, args, contexts_to_process), join=True)
    else:
        print("No new contexts left. Merging existing output and temporary shards...")

    processed, matched, failed, total_rows = merge_part_files(
        args,
        args.run_tag if contexts_to_process else None,
        world_size,
        output_jsonl_path,
    )

    if os.path.isdir(args.tmp_dir) and not os.listdir(args.tmp_dir):
        os.rmdir(args.tmp_dir)

    match_rate = (matched / processed) if processed else 0.0

    print(f"Stego text saved to: {output_jsonl_path}")
    print(f"Total rows in output JSONL: {total_rows}")
    print(f"Processed: {processed}, Matched: {matched}, Failed: {failed}, Extract match rate: {match_rate:.4f}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
