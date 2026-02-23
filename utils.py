import torch
import torch.nn.functional as F
import numpy as np
import bitarray
import random
from typing import List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, Cache, DynamicCache
from config import resolve_model_path as config_resolve_model_path


def resolve_model_path(model_name: str, model_path: str = "") -> str:
    # Backward-compatible wrapper.
    return config_resolve_model_path(model_name, model_path)


def random_bits(length: int) -> str:
    return "".join(random.choice("01") for _ in range(length))


def normalize_bits(bits_obj) -> str:
    if bits_obj is None:
        return ""
    if isinstance(bits_obj, str):
        return bits_obj
    if isinstance(bits_obj, (list, tuple)):
        return "".join(str(x) for x in bits_obj)
    return str(bits_obj)


def initialize_model_and_precision(args, seed: int):
    args.model_path = resolve_model_path(args.model_name, args.model_path)
    enc, model, precision_int = get_model(
        model_name=args.model_path,
        model_precision=args.model_precision,
        seed=seed,
    )
    if args.ac_precision is None:
        args.ac_precision = precision_int
    return enc, model


def encode_with_method(method: str, model, context_tokens: List[int], message_bits: str, args, device: str) -> Tuple[List[int], str]:
    # Delayed imports avoid circular dependency with pss modules importing utils.
    if method == "ac":
        from pss.arithmetic import encode_arithmetic

        generated_ids, encoded_message, *_ = encode_arithmetic(
            model,
            message_bits,
            context_tokens,
            args.max_tokens,
            device=device,
            precision=args.ac_precision,
            top_p=args.top_p,
            top_k=args.top_k,
            temp=args.temp,
        )
    elif method == "meteor":
        from pss.meteor import encode_meteor

        generated_ids, encoded_message, *_ = encode_meteor(
            model,
            context_tokens,
            message_bits,
            args.max_tokens,
            device=device,
            top_p=args.top_p,
            top_k=args.top_k,
            precision=args.ac_precision,
            reorder=args.meteor_reorder,
            temp=args.temp,
        )
    elif method == "adg":
        from pss.adg import encode_adg

        generated_ids, encoded_message, *_ = encode_adg(
            model,
            context_tokens,
            message_bits,
            args.max_tokens,
            device=device,
            top_p=args.top_p,
            top_k=args.top_k,
            temp=args.temp,
        )
    elif method == "discop":
        from pss.discop import encode_discop

        generated_ids, encoded_message, *_ = encode_discop(
            model,
            context_tokens,
            message_bits,
            args.max_tokens,
            args.seed,
            device=device,
            top_p=args.top_p,
            top_k=args.top_k,
            baseline_flag=args.discop_baseline,
            temp=args.temp,
        )
    elif method == "imec":
        from pss.imec import encode_imec

        generated_ids, encoded_message, *_ = encode_imec(
            model,
            context_tokens,
            message_bits,
            args.max_tokens,
            args.seed,
            block_size=args.imec_block_size,
            device=device,
            top_p=args.top_p,
            top_k=args.top_k,
            temp=args.temp,
        )
    elif method == "sparsamp":
        from pss.sparsamp import encode_spar

        generated_ids, encoded_message, *_ = encode_spar(
            model,
            context_tokens,
            message_bits,
            args.max_tokens,
            device=device,
            block_size=args.sparsamp_block_size,
            top_p=args.top_p,
            top_k=args.top_k,
            random_seed=args.seed,
            temp=args.temp,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    return list(generated_ids), normalize_bits(encoded_message)


def decode_with_method(method: str, model, context_tokens: List[int], generated_ids: List[int], args, device: str) -> str:
    # Delayed imports avoid circular dependency with pss modules importing utils.
    if method == "ac":
        from pss.arithmetic import decode_arithmetic

        extracted_bits = decode_arithmetic(
            model,
            generated_ids,
            context_tokens,
            device=device,
            precision=args.ac_precision,
            top_p=args.top_p,
            top_k=args.top_k,
            temp=args.temp,
        )
    elif method == "meteor":
        from pss.meteor import decode_meteor

        extracted_bits = decode_meteor(
            model,
            generated_ids,
            context_tokens,
            device=device,
            top_p=args.top_p,
            top_k=args.top_k,
            precision=args.ac_precision,
            reorder=args.meteor_reorder,
            temp=args.temp,
        )
    elif method == "adg":
        from pss.adg import decode_adg

        extracted_bits = decode_adg(
            model,
            generated_ids,
            context_tokens,
            device=device,
            top_p=args.top_p,
            top_k=args.top_k,
            temp=args.temp,
        )
    elif method == "discop":
        from pss.discop import decode_discop

        extracted_bits = decode_discop(
            model,
            context_tokens,
            generated_ids,
            args.seed,
            device=device,
            top_p=args.top_p,
            top_k=args.top_k,
            baseline_flag=args.discop_baseline,
            temp=args.temp,
        )
    elif method == "imec":
        from pss.imec import decode_imec

        extracted_bits = decode_imec(
            model,
            context_tokens,
            generated_ids,
            args.imec_block_size,
            device=device,
            top_p=args.top_p,
            top_k=args.top_k,
            temp=args.temp,
        )
    elif method == "sparsamp":
        from pss.sparsamp import decode_spar

        extracted_bits = decode_spar(
            model,
            generated_ids,
            context_tokens,
            device=device,
            block_size=args.sparsamp_block_size,
            top_p=args.top_p,
            top_k=args.top_k,
            random_seed=args.seed,
            temp=args.temp,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    return normalize_bits(extracted_bits)

# def decode(self, token_ids, **kwargs):
#     filtered_tokens = self.convert_ids_to_tokens(token_ids)
#     text = self.convert_tokens_to_string(filtered_tokens)
#     return text
# AutoTokenizer.decode = decode

def _convert_token_to_id(self, token):
    return self.encoder.get(token, 0)
AutoTokenizer._convert_token_to_id = _convert_token_to_id


# handles both old and new cache formats
def limit_past(past):
    if isinstance(past, Cache):
        for i in range(len(past.key_cache)):
            past.key_cache[i] = past.key_cache[i][:, :, -1022:]
            past.value_cache[i] = past.value_cache[i][:, :, -1022:]
        return past

    if past is None:
        return None

    past = list(past)
    for i in range(len(past)):
        if isinstance(past[i], tuple):
            key, value = past[i]
            past[i] = (
                key[:, :, -1022:],
                value[:, :, -1022:]
            )
        else:
            past[i] = past[i][:, :, -1022:]
    return tuple(past)

def kl(q, logq, logp):
    res = q*(logq-logp)/0.69315
    res[q==0] = 0
    return res.sum().item() # in bits

def entropy(q, logq):
    res = q*logq/0.69315
    res[q==0] = 0
    return -res.sum().item() # in bits


def _past_seq_len(past):
    if past is None:
        return 0

    if isinstance(past, Cache):
        if len(past.key_cache) == 0:
            return 0
        # key_cache shape: [batch, heads, seq_len, head_dim]
        return int(past.key_cache[0].shape[-2])

    if isinstance(past, (tuple, list)) and len(past) > 0:
        first = past[0]
        if isinstance(first, tuple):
            # legacy cache item: (key, value)
            return int(first[0].shape[-2])
        # fallback tensor-like cache
        return int(first.shape[-2])

    return 0


def sample(model, enc, length, context, temperature=1.0, device='cuda', topk=-1):
    """
    Baseline sampling helper (migrated from sample.py).
    Returns:
      generated_token_ids, avg_nll, avg_kl, avg_entropy
    """
    assert length > 0

    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    prev = context
    output = context
    past = None

    total_log_probs = 0.0
    total_entropy_ptau = 0.0
    total_num = 0
    total_kl = 0.0  # in bits

    with torch.no_grad():
        while total_num < length:
            if _past_seq_len(past) >= 1023:
                raise RuntimeError("Past key values exceed supported context window.")

            if prev.dim() == 1:
                prev_in = prev.unsqueeze(0)
            else:
                prev_in = prev

            if past is None:
                model_output = model(prev_in, use_cache=True)
            else:
                try:
                    model_output = model(prev_in, past_key_values=past, use_cache=True)
                except ValueError as e:
                    msg = str(e)
                    if "past_key_values" in msg and "Cache" in msg and not isinstance(past, Cache):
                        model_output = model(prev_in, past_key_values=DynamicCache.from_legacy_cache(past), use_cache=True)
                    else:
                        raise

            logits = model_output.logits
            past = limit_past(model_output.past_key_values)

            next_token_logits = logits[0, -1, :]
            # Keep legacy behavior used by this repository.
            next_token_logits[-1] = -1e10
            if next_token_logits.numel() > 628:
                next_token_logits[628] = -1e10

            sorted_logits, indices = next_token_logits.sort(descending=True)
            base_log_probs = F.log_softmax(sorted_logits, dim=-1)

            if topk > 0:
                sorted_logits = sorted_logits[:topk]
                indices = indices[:topk]
                base_log_probs_for_kl = base_log_probs[:topk]
            else:
                base_log_probs_for_kl = base_log_probs

            sorted_logits = sorted_logits / temperature
            log_probs = F.log_softmax(sorted_logits, dim=-1)
            probs = torch.exp(log_probs)

            total_kl += kl(probs, log_probs, base_log_probs_for_kl)

            selection = torch.multinomial(probs, num_samples=1).item()
            total_log_probs += float(base_log_probs_for_kl[selection])
            total_entropy_ptau += entropy(probs, log_probs)

            prev = indices[selection].view(1)
            output = torch.cat((output, prev))
            total_num += 1

    avg_nll = -total_log_probs / total_num
    avg_kl = total_kl / total_num
    avg_hq = total_entropy_ptau / total_num

    return output[len(context):].tolist(), avg_nll, avg_kl, avg_hq

# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit*(2**i)
    return res

def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}'%num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]

def is_sent_finish(token_idx, enc):
    token = enc.decode([token_idx])
    return '.' in token or '!' in token or '?' in token

def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i
    #         return i
    # return len(bits1)

def encode_context(raw_text, enc):
    # context_tokens = enc.encode('<|endoftext|>') + enc.encode(raw_text)
    context_tokens = enc.encode(raw_text)
    return context_tokens

# Use gpt2-medium for 345M param model
# Use gpt2-large for 774M param model
def get_model(model_name='gpt2', device=None, seed=1234, model_precision='float32'):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dtype_map = {
        'float32': (torch.float32, 32),
        'float16': (torch.float16, 16),
    }
    torch_dtype, precision_int = dtype_map.get(model_precision, (torch.float32, 32))
    
    if model_precision not in dtype_map:
        print(f"Warning: Invalid model_precision '{model_precision}'. Defaulting to float32.")

    model = None
    enc = None
    
    if 'gpt2' in model_name:
        enc = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # enc.unk_token = None
        # enc.bos_token = None
        # enc.eos_token = None
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True)
    elif 'Qwen' in model_name:
        enc = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True)
    elif '9G4B' in model_name:
        enc = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True)
    elif 'Llama' in model_name:
        enc = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True)
    
    # if model is not None:
    #     model.to(device)
    #     model.eval()
    else:
        raise ValueError(f"Model name '{model_name}' not recognized or supported.")
    
    # model.double() is removed as precision is handled by torch_dtype

    return enc, model, precision_int

enc32_itoc = ['\0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '.', ',', "'", '!', ' ']
enc32_ctoi = {k: v for v, k in enumerate(enc32_itoc)}
def enc32(text):
    bits = []
    for c in text:
        bits.extend(int2bits(enc32_ctoi[c], 5))
    return bits

def dec32(bits):
    text = ''
    for i in range(0, len(bits), 5):
        c = enc32_itoc[bits2int(bits[i:i+5])]
        if c == '\0':
            break
        text += c
    return text

# message should be bit string
# encoded should be text string
def expansion_ratio(message, encoded):
    message_bits = len(message)
    encoded_ba = bitarray.bitarray()
    encoded_ba.frombytes(encoded.encode('utf-8'))
    encoded_bits = len(encoded_ba.tolist())
    return encoded_bits/message_bits

from collections import namedtuple

# Data structure for holding the output of a single encoding example
SingleExampleOutput = namedtuple('SingleExampleOutput', [
    'generated_ids', 
    'unknown1',  # Placeholder for compatibility
    'n_bits', 
    'total_entropy', 
    'ave_kld', 
    'max_kld', 
    'perplexity', 
    'time_cost', 
    'settings', 
    'unknown2',  # Placeholder for compatibility
    'ave_kld_trunc', 
    'max_kld_trunc', 
    'total_model_time', 
    'total_sample_time'
])

def filter_logits(logits, top_p=0.9):
    """
    Nucleus sampling filter for logits.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float('Inf')
    return logits

def filter_top_k(logits, top_k):
    """
    Top-K sampling filter for logits.
    只保留概率最大的 K 个 token，其余设为 -inf
    """
    if top_k > 0:
        top_k_values = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        indices_to_remove = logits < top_k_values
        logits[indices_to_remove] = -float('Inf')
        
    return logits

def get_probs_past(model, prev=None, past=None, device='cuda', top_p=1.0, top_k=0, temp=1.0):
    """
    A new version of get_probs_past that accepts a temperature parameter.
    """
    if prev.dim() == 1:
        prev = prev.unsqueeze(0)
    
    if past is None:
        model_output = model(prev, use_cache=True)
    else:
        try:
            model_output = model(prev, past_key_values=past, use_cache=True)
        except ValueError as e:
            msg = str(e)
            if "past_key_values" in msg and "Cache" in msg and not isinstance(past, Cache):
                model_output = model(prev, past_key_values=DynamicCache.from_legacy_cache(past), use_cache=True)
            else:
                raise
    past = model_output.past_key_values

    logits = model_output.logits[0, -1, :].to(device)
    logits, indices = logits.sort(descending=True)
    logits = logits.double()
    indices = indices.int()
    
    # Apply temperature
    logits = logits / temp
    probs = F.softmax(logits, dim=-1)

    if top_k > 0:
        probs = probs[:top_k]
        indices = indices[:top_k]
        probs = probs / probs.sum()
        
    if 0 < top_p < 1.0:
        cum_probs = probs.cumsum(0)
        # Ensure we don't get an out-of-bounds error if top_p is very high
        k_tensor = (cum_probs > top_p).nonzero()
        if k_tensor.numel() > 0:
            k = k_tensor[0].item() + 1
            probs = probs[:k]
            indices = indices[:k]
            probs = 1 / cum_probs[k - 1] * probs  # Normalizing
        
    return probs, indices, past


def get_lower_upper_bound(cumulative_probs, v):
    lower_bound = cumulative_probs[v-1] if v > 0 else torch.tensor(0)
    upper_bound = cumulative_probs[v] if v < len(cumulative_probs)-1 else torch.tensor(1)
    SE = [lower_bound.item(), upper_bound.item()]
    return SE

def func_mrn(k_m, n_m, r):
    result = ((k_m / n_m) + r)
    if result >= 1:
        result = result - 1
    return result

def dec2bin(km, lm):
    bin_str = bin(km)[2:]
    return bin_str.zfill(lm)
