import bitarray
import numpy as np
from utils import get_probs_past
from copy import deepcopy
import torch
from scipy.stats import entropy
import hashlib
import hmac
import time
from scipy.stats import entropy


def greatest_lower_bound(p, q):
    """
  Calculate the greatest lower bound of two distributions p, q using
  cicalese_supermodularity_2002 - Definition 3
  Note: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=992785 FACT 1 is incorrect - see above!
  """
    if p.shape[0] < q.shape[0]:
        p = np.concatenate([p, np.zeros(q.shape[0] - p.shape[0])])
    elif q.shape[0] < p.shape[0]:
        q = np.concatenate([q, np.zeros(p.shape[0] - q.shape[0])])
    p_cumsum = np.cumsum(p, dtype=np.float64)
    q_cumsum = np.cumsum(q, dtype=np.float64)
    z = np.minimum(p_cumsum, q_cumsum, dtype=np.float64)
    z[1:] -= np.minimum(p_cumsum[:-1], q_cumsum[:-1], dtype=np.float64)
    return z

def entropy2(q, prec=18):
    res = q * np.log2(q)
    res[q == 0] = 0
    ressum = res.sum()
    return -np.around(ressum, decimals=prec)


class DRBG(object):
    def __init__(self, key, seed):
        self.key = key
        self.val = b'\x01' * 64
        self.reseed(seed)

        self.byte_index = 0
        self.bit_index = 0

    def hmac(self, key, val):
        return hmac.new(key, val, hashlib.sha512).digest()

    def reseed(self, data=b''):
        self.key = self.hmac(self.key, self.val + b'\x00' + data)
        self.val = self.hmac(self.key, self.val)

        if data:
            self.key = self.hmac(self.key, self.val + b'\x01' + data)
            self.val = self.hmac(self.key, self.val)

    def generate_bits(self, n):
        xs = np.zeros(n, dtype=bool)
        for i in range(0, n):
            xs[i] = (self.val[self.byte_index] >> (7 - self.bit_index)) & 1

            self.bit_index += 1
            if self.bit_index >= 8:
                self.bit_index = 0
                self.byte_index += 1

            if self.byte_index >= 8:
                self.byte_index = 0
                self.val = self.hmac(self.key, self.val)

        self.reseed()
        return xs



# Constants for HMAC-DRBG -- MUST CHANGE FOR SECURE IMPLEMENTATION
sample_key = b'0x01' * 64
sample_seed_prefix = b'sample'
sample_nonce_counter = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'


def mec_kocaoglu_np(p: np.array, q: np.array):
    """
    Algorithm 1:  https://arxiv.org/pdf/1611.04035.pdf
    We adjust Algorithm 1 and follow the advice in the text in order to reconstruct the matrix.

    Supposedly has 1-bit guarantee - unfortunately not clear if equal to kacaoglu2

    We require len(p) == q.
    """
    p = p.copy().astype(np.float64)
    p /= p.sum()
    q = q.copy().astype(np.float64)
    q /= q.sum()
    assert len(p) == len(q), "len(p) must be equal to len(q)!"
    J = np.zeros((q.shape[0], p.shape[0]), dtype=np.float64)  # Joint distribution

    # e = []
    M = np.stack((p, q), 0)
    r = M.max(axis=1).min()
    while r > 0:
        # e.append(r)
        a_i = M.argmax(axis=1)
        M[0, a_i[0]] -= r
        M[1, a_i[1]] -= r
        J[a_i[0], a_i[1]] = r
        r = M.max(axis=1).min()
    return J

def minimum_entropy_coupling(p: np.ndarray, q: np.ndarray, method="kocaoglu", select_col="all", select_row="all",
                             verbose=False, **kwargs):
    global mec_cpp
    assert p.ndim == 1 and q.ndim == 1, "ERROR: batch mode not yet supported!"
    stats = {}
    ret_dct = {}

    if not (np.isclose(p.sum(), np.float64(1.0)) and
            np.isclose(np.sum(p[p < 0.0]), np.float64(0.0)) and
            np.isclose(q.sum(), np.float64(1.0)) and
            np.isclose(np.sum(q[q < 0.0]), np.float64(0.0))):
        assert False, "Either q, p (or both) are not proper probability distributions! p: {} q: {}".format(p, q)

    if p.shape[0] > q.shape[0]:
        q = np.concatenate([q, np.zeros(p.shape[0] - q.shape[0], dtype=np.float64)])
    elif q.shape[0] > p.shape[0]:
        p = np.concatenate([p, np.zeros(q.shape[0] - p.shape[0], dtype=np.float64)])

    p = p.astype(np.float64)
    p /= p.sum()
    q = q.astype(np.float64)
    q /= q.sum()

    # Note: We here use a numerically stabilised variant of MEC by Koacoglu et al, 2017
    # Runtime complexity is O(max(p_dim, q_dim)**2)
    M = mec_kocaoglu_np(p, q)

    q_est = M.sum(0)
    q_est = q_est / q_est.sum()

    # ret_dct["q_entropy"] = entropy(q, base=2)
    # ret_dct["kl_q"] = entropy(q, q_est, base=2)
    # if not np.isfinite(ret_dct["kl_q"]):
    #     ret_dct["kl_q"] = entropy(q + 1e-30, q_est + 1e-30, base=2)
    # ret_dct["stat_time"] = time.time() - t1

    if select_row not in [None, "all"]:
        ret_dct["M_selected_row"] = M[select_row]
    if select_col not in [None, "all"]:
        ret_dct["M_selected_col"] = M[:, select_col]
    ret_dct["M_colfirst"] = np.transpose(M)
    return ret_dct


def apply_random_mask(message_bits, input_key, sample_seed_prefix, input_nonce):
    mask_generator = DRBG(input_key, sample_seed_prefix + input_nonce)
    mask_bits = mask_generator.generate_bits(len(message_bits))
    masked_message_bits = deepcopy(message_bits)
    for b in range(0, len(message_bits)):
        masked_message_bits[b] = message_bits[b] ^ mask_bits[b]
    return masked_message_bits


def remove_random_mask(message_bits, input_key, sample_seed_prefix, input_nonce):
    return apply_random_mask(message_bits, input_key, sample_seed_prefix, input_nonce)



@torch.no_grad()
def encode_imec(model, context, message, token_num_need_generated, seed, block_size=10, device='cuda', top_p=1.0, top_k=0, pad_last_belief_chunk=True, belief_entropy_threshold=1E-9, temp=1.0):
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    g = np.random.default_rng(seed)

    past = None
    prev = context
    generated_ids = []
    encoded_message = []
    token_num_generated = 0
    num_bits_encoded = 0
    m_index = 0
    msg_per_coup = 80
    total_entropy = 0
    stat_time = 0
    model_time = 0
    
    message_len = len(message)
    if message_len % msg_per_coup != 0:
        padding_len = msg_per_coup - (message_len % msg_per_coup)
        message += '0' * padding_len
    message_len = len(message)


    while m_index < message_len:
        message_bits = message[m_index: m_index + msg_per_coup]
        if not message_bits:
            break
        message_bits = list(int(x) for x in message_bits)

        # caculate chunks
        block_sizes = [block_size for i in range(int(len(message_bits) // block_size))]
        if len(message_bits) % block_size:
            block_sizes += [int(len(message_bits) % block_size)]
        idx = 0
        msg_chunks = []
        for cs in block_sizes:
            msg_chunk = np.array(message_bits[idx: idx + cs]).dot(
                1 << np.arange(cs, dtype='int64')[::-1]
            )
            msg_chunks.append(msg_chunk)
            idx += cs

        beliefs = [np.zeros(2 ** cs, dtype=np.float64) + 1.0 / (2 ** cs) for cs in block_sizes]

        belief_entropies = np.array([entropy2(b) for b in beliefs])
        n_steps = 0
        is_encoded = False
        while True:
            if token_num_generated >= token_num_need_generated:
                # print(f"Warning: Token limit ({token_num_need_generated}) reached, but message not fully encoded. Encoded {m_index} of {message_len} bits.")
                break
            model_time_1 = time.time()
            probs, indices, past = get_probs_past(model, prev, past, device, top_p, top_k, temp)
            model_time_2 = time.time()
            model_time += model_time_2 - model_time_1

            stat_time_1 = time.time()
            entropy_t = entropy(probs.cpu(),base=2)
            total_entropy += entropy_t
            stat_time_2 = time.time()
            stat_time += stat_time_2 - stat_time_1

            n_steps += 1
            assert n_steps < len(message_bits) * 15, "Error: iMEC seems to be stuck in a loop, skip this one."

            next_chunk_id = np.argmax(belief_entropies)
            next_chunk_content = msg_chunks[next_chunk_id]

            mec_dict = minimum_entropy_coupling(
                beliefs[next_chunk_id],
                probs.cpu().numpy(),
                select_row = next_chunk_content,
                select_col = "all",
                mode = "dense",
                algo_atol= 1E-7,
                warning_atol = 1E-5
            )

            M_row_next_chunk = mec_dict["M_selected_row"]
            M_row_next_chunk = M_row_next_chunk / M_row_next_chunk.sum()
            next_action = g.choice(M_row_next_chunk.shape[0], p=M_row_next_chunk)
            belief_update = mec_dict['M_colfirst'][next_action]
            beliefs[next_chunk_id] = belief_update / belief_update.sum()
            belief_entropies[next_chunk_id] = entropy(beliefs[next_chunk_id], base=2)
            sampled_index = indices[next_action].item()
            generated_ids.append(sampled_index)
            prev = torch.tensor([sampled_index], device=device).unsqueeze(0)
            token_num_generated += 1
            if max(belief_entropies) <= belief_entropy_threshold:
                is_encoded = True
                break
        
        if not is_encoded:
            break

        num_bits_encoded += len(message_bits)
        encoded_message.append("".join(map(str, message_bits)))
        m_index += msg_per_coup

    return generated_ids, encoded_message, total_entropy, stat_time, model_time


@torch.no_grad()
def decode_imec(model, context, generated_ids, block_size, device='cuda',top_p=1.0, top_k=0, pad_last_belief_chunk=True, belief_entropy_threshold=1E-9, temp=1.0 ):
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    past = None
    prev = context
    decoded_message = []
    message_length_per_cop = 80

    # Since the encoder now pads all messages, every segment is a full 80 bits.
    # This means the block_sizes array is constant for every segment.
    block_sizes = [block_size] * (message_length_per_cop // block_size)
    beliefs = [np.zeros(2 ** block_size, dtype=np.float64) + 1.0 / (2 ** block_size) 
               for _ in block_sizes]

    for tokenID in generated_ids:
        probs, indices, past = get_probs_past(model, prev, past, device, top_p, top_k, temp)
        belief_entropies = np.array([entropy2(b) for b in beliefs])

        next_chunk_id = np.argmax(belief_entropies)
        
        try:
            next_action = indices.cpu().tolist().index(tokenID)
        except ValueError:
            # This can happen if top_p/top_k settings differ, or if the model is highly non-deterministic
            # in ways not captured by the random seed. We'll skip the belief update for this token.
            prev = torch.tensor([tokenID], device=device).unsqueeze(0)
            continue

        mec_dict = minimum_entropy_coupling(
            beliefs[next_chunk_id],
            probs.cpu().numpy(),
            select_row=None,
            select_col=next_action,
            method='kacaoglu',
            mode="dense",
            algo_atol=1E-7,
            warning_atol=1E-5
        )
        vec2 = mec_dict['M_selected_col']
        prev = torch.tensor([tokenID], device=device).unsqueeze(0)

        # Update belief, guarding against division by zero.
        if vec2.sum() > 1e-9:
            beliefs[next_chunk_id] = vec2 / vec2.sum()
        
        # Check if the current segment is fully decoded
        belief_entropies[next_chunk_id] = entropy2(beliefs[next_chunk_id])
        if max(belief_entropies) <= belief_entropy_threshold:
            output = [format(np.argmax(b), '0{}b'.format(cs)) for b, cs in zip(beliefs, block_sizes)]
            output = "".join(output)
            decoded_message.append(output)
            
            # Reset beliefs for the next segment
            beliefs = [np.zeros(2 ** block_size, dtype=np.float64) + 1.0 / (2 ** block_size) 
                       for _ in block_sizes]

    return decoded_message

# 11011110111100111000111111101100000000011100111110011111101111011111010110011000
# 10110010101100011100001000100011101110100011000011100111100110010110100010110001



@torch.no_grad()
def encode_imec_single(model, context, message, token_num_need_generated, seed, block_size=10, device='cuda', top_p=1.0, pad_last_belief_chunk=True, belief_entropy_threshold=1E-9, temp=1.0):
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    g = np.random.default_rng(seed)

    past = None
    prev = context
    generated_ids = []
    encoded_message = []
    token_num_generated = 0
    num_bits_encoded = 0
    m_index = 0

    i = 0
    while token_num_generated < token_num_need_generated:
        msg_chunk = int(message[m_index:m_index + block_size], 2)

        belief = np.zeros(2 ** block_size, dtype=np.float64) + 1.0 / (2 ** block_size)
        belief_entropy = entropy(belief, base=2)

        cur_token_id = []
        i = 0
        while True:
            probs, indices, past = get_probs_past(model, prev, past, device, top_p, temp)
            next_chunk_content = msg_chunk
            mec_dict = minimum_entropy_coupling(
                belief,
                probs.cpu().numpy(),
                select_row=next_chunk_content,
                select_col="all",
                mode="dense",
                algo_atol=1E-7,
                warning_atol=1E-5
            )
            M_row_next_chunk = mec_dict["M_selected_row"]
            M_row_next_chunk = M_row_next_chunk / M_row_next_chunk.sum()
            next_action = g.choice(M_row_next_chunk.shape[0], p=M_row_next_chunk)
            belief_update = mec_dict['M_colfirst'][next_action]
            belief = belief_update / belief_update.sum()
            belief_entropy = entropy(belief,base=2)
            sampled_index = indices[next_action].item()
            generated_ids.append(sampled_index)
            cur_token_id.append(sampled_index)
            prev = torch.tensor([sampled_index], device=device).unsqueeze(0)
            token_num_generated += 1
            i += 1

            if belief_entropy <= belief_entropy_threshold:
                cur_token_id = []
                is_encoded = True
                break

        num_bits_encoded += block_size
        encoded_message.append(message[m_index : m_index+block_size])
        m_index += block_size

    return generated_ids, encoded_message


@torch.no_grad()
def decode_imec_single(model, context, generated_ids, block_size=10, device='cuda', top_p=1.0, pad_last_belief_chunk=True, belief_entropy_threshold=1E-9, temp=1.0):
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    past = None
    prev = context
    decoded_message = []
    belief = np.zeros(2 ** block_size, dtype=np.float64) + 1.0 / (2 ** block_size)

    cur_tokenID = []
    i = 0
    for tokenID in generated_ids:
        cur_tokenID.append(tokenID)
        probs, indices, past = get_probs_past(model, prev, past, device, top_p, temp)
        belief_entropy = entropy(belief, base=2)
        next_action = indices.cpu().tolist().index(tokenID)

        mec_dict = minimum_entropy_coupling(
            belief,
            probs.cpu().numpy(),
            select_row='all',
            select_col=next_action,
            method='kacaoglu',
            mode="dense",
            algo_atol=1E-7,
            warning_atol=1E-5
        )
        vec2 = mec_dict['M_selected_col']
        prev = torch.tensor([tokenID], device=device).unsqueeze(0)
        belief = vec2 / vec2.sum()
        belief_entropy = entropy(belief,base=2)
        i += 1
        if belief_entropy <= belief_entropy_threshold:
            cur_tokenID = []
            output = format(np.argmax(belief), '0{}b'.format(block_size))
            decoded_message.append(output)
            belief = np.zeros(2 ** block_size, dtype=np.float64) + 1.0 / (2 ** block_size)

    return decoded_message
