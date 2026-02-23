"""Sparse sampling steganography encode and decode implementations."""

import random

import torch
from math import ceil
from utils import func_mrn, dec2bin, get_lower_upper_bound, get_probs_past


def encode_step(probs, n_m, k_m):
    """Execute one encoding step for the current distribution."""
    r = random.random()
    cumulative_probs = probs.cumsum(0)
    r_i_m = func_mrn(k_m, n_m, r)
    token_index = (cumulative_probs > r_i_m).nonzero()[0].item()

    SE = get_lower_upper_bound(cumulative_probs, token_index)
    temp0 = ceil((SE[0] - r) * n_m)
    temp1 = ceil((SE[1] - r) * n_m)

    if k_m + r * n_m >= n_m:
        k_m = k_m - n_m - temp0
    else:
        k_m = k_m - temp0
    n_m = temp1 - temp0

    return token_index, n_m, k_m


import time
from scipy.stats import entropy


@torch.no_grad()
def encode_spar(
    model,
    context,
    message_bits,
    token_num_need_generation,
    device="cuda",
    block_size=32,
    top_p=1.0,
    top_k=0,
    random_seed=42,
    temp=1.0,
):
    """Encode message bits using sparse sampling."""
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    generated_ids = []
    m_index = 0
    bits_slice = message_bits[:block_size]
    if len(bits_slice) < block_size:
        bits_slice = bits_slice.ljust(block_size, "0")
    k_m = int(bits_slice, 2)
    n_m = 2**block_size
    token_num_generated = 0
    random.seed(random_seed)
    encoded_message = []
    past = None
    prev = context
    total_entropy = 0
    stat_time = 0
    model_time = 0
    message_len = len(message_bits)

    while m_index < message_len:
        if token_num_generated >= token_num_need_generation:
            break

        model_time_1 = time.time()
        probs, indices, past = get_probs_past(
            model=model,
            prev=prev,
            past=past,
            device=device,
            top_p=top_p,
            top_k=top_k,
            temp=temp,
        )
        model_time_2 = time.time()
        model_time += model_time_2 - model_time_1

        stat_time_1 = time.time()
        entropy_t = entropy(probs.cpu(), base=2)
        total_entropy += entropy_t
        stat_time_2 = time.time()
        stat_time += stat_time_2 - stat_time_1

        probs = probs.to(torch.float64)
        token_index, n_m, k_m = encode_step(probs=probs, n_m=n_m, k_m=k_m)
        tokenID = indices[token_index]
        token_num_generated += 1
        generated_ids.append(tokenID.item())

        if n_m == 1:
            encoded_message.append(message_bits[m_index : m_index + block_size])
            m_index += block_size

            if m_index < message_len:
                n_m = 2**block_size
                bits_slice = message_bits[m_index : m_index + block_size]
                if len(bits_slice) < block_size:
                    bits_slice = bits_slice.ljust(block_size, "0")
                k_m = int(bits_slice, 2)

        prev = torch.tensor([tokenID], device=device, dtype=torch.long).unsqueeze(0)

    return generated_ids, encoded_message, total_entropy, stat_time, model_time


@torch.no_grad()
def encode_spar2(
    model,
    context,
    message_bits,
    token_num_need_generation,
    device="cuda",
    block_size=32,
    top_p=1.0,
    top_k=0,
    random_seed=42,
    temp=1.0,
):
    """Run the legacy sparse-sampling encoder variant."""
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    generated_ids = []
    m_index = 0
    k_m = int(message_bits[:block_size], 2)
    n_m = 2**block_size
    token_num_generated = 0
    random.seed(random_seed)
    encoded_message = []
    past = None
    prev = context
    total_entropy = 0
    stat_time = 0
    model_time = 0

    while m_index < 1023:
        model_time_1 = time.time()
        probs, indices, past = get_probs_past(
            model=model,
            prev=prev,
            past=past,
            device=device,
            top_p=top_p,
            top_k=top_k,
            temp=temp,
        )
        model_time_2 = time.time()
        model_time += model_time_2 - model_time_1

        stat_time_1 = time.time()
        entropy_t = entropy(probs.cpu(), base=2)
        total_entropy += entropy_t
        stat_time_2 = time.time()
        stat_time += stat_time_2 - stat_time_1

        probs = probs.to(torch.float64)
        token_index, n_m, k_m = encode_step(probs=probs, n_m=n_m, k_m=k_m)
        tokenID = indices[token_index]
        if n_m == 1:
            encoded_message.append(message_bits[m_index : m_index + block_size])
            m_index += block_size
            n_m = 2**block_size
            k_m = int(message_bits[m_index : m_index + block_size], 2)
        token_num_generated += 1
        generated_ids.append(tokenID.item())
        if len(generated_ids) > 12000:
            print(
                f"We have generated more than 12000 tokens,but this block message still not embedded over. This context seems have problem. let's skip it."
            )
            raise Exception("This context seems have problem.let's skip it.")

        prev = torch.tensor([tokenID], device=device, dtype=torch.long).unsqueeze(0)

    return generated_ids, encoded_message, total_entropy, stat_time, model_time


@torch.no_grad()
def decode_spar(
    model,
    generated_ids,
    context,
    device="cuda",
    block_size=32,
    top_p=1.0,
    top_k=0,
    random_seed=42,
    temp=1.0,
):
    """Decode message bits from sparse-sampling token IDs."""
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    random.seed(random_seed)
    message = []
    n_m = 2**block_size
    k_m = 0
    n_m_arr = []
    temp0_arr = []
    temp1_arr = []
    past = None
    prev = context

    for tokenID in generated_ids:
        r = random.random()
        probs, indices, past = get_probs_past(
            model=model,
            prev=prev,
            past=past,
            device=device,
            top_p=top_p,
            top_k=top_k,
            temp=temp,
        )
        probs = probs.to(torch.float64)
        cumulative_probs = probs.cumsum(0)

        # 这里需要根据tokenID在indices中获取下标
        token_index = torch.where(indices == tokenID)[0]
        SE = get_lower_upper_bound(cumulative_probs, token_index)

        temp0 = ceil((SE[0] - r) * n_m)
        temp1 = ceil((SE[1] - r) * n_m)

        n_m = temp1 - temp0
        temp0_arr.append(temp0)
        temp1_arr.append(temp1)
        n_m_arr.append(n_m)

        if n_m == 1:
            count = len(temp0_arr) - 2
            k_m = temp0_arr[count + 1]
            while count >= 0:
                n_m_new = n_m_arr[count]
                k_m = temp0_arr[count] + ((k_m + n_m_new) % n_m_new)
                count -= 1
            k_m = (k_m + 2**block_size) % 2**block_size
            temp0_arr = []
            temp1_arr = []
            n_m_arr = []
            message.append(dec2bin(k_m, block_size))
            n_m = 2**block_size
        prev = torch.tensor([tokenID], device=device, dtype=torch.long).unsqueeze(0)

    return message
