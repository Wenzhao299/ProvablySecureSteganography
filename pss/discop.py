import torch
from math import log2
from collections import deque
from utils import get_probs_past,bits2int
from typing import List, Dict, Optional, Union, Tuple
import random

class Node:
    # node of tree
    def __init__(self,
                 prob: float,
                 left=None,
                 right=None,
                 *,
                 index: Optional[int] = None,
                 search_path: Optional[int] = None,
                 label: Optional[str] = None) -> None:
        # `search_path is None` - no need
        # `search_path == 0` - self
        # `search_path == -1` - left subtree
        # `search_path == 1` - right subtree

        self.left = left
        self.right = right
        self.prob = prob
        self.index = index
        self.search_path = search_path
        self.label = str(index) if index is not None else label

    def __lt__(self, c) -> bool:
        if isinstance(c, float):
            return self.prob < c
        return self.prob < c.prob

    def __le__(self, c) -> bool:
        if isinstance(c, float):
            return self.prob <= c
        return self.prob <= c.prob

    def is_leaf(self) -> bool:
        return self.index is not None


def create_huffman_tree(indices: List[int], probs: List[float], search_for: Optional[int] = None) -> Tuple[Node, float]:
    sz = len(indices)
    nodes = [Node(probs[i], index=indices[i], search_path=(0 if search_for == indices[i] else None)) for i in range(sz)]
    # nodes.sort()  # maybe already sorted
    # nodes = nodes.reverse()
    a = deque(nodes)
    b = deque()

    def fun():
        nonlocal a, b
        if len(a) > 0 and len(b) > 0 and a[-1] <= b[-1]:
            item = a.pop()
        elif len(a) == 0:
            item = b.pop()
        elif len(b) == 0:
            item = a.pop()
        else:
            item = b.pop()
        return item

    while len(a) + len(b) > 1:
        left = fun()
        right = fun()
        prob = left.prob + right.prob
        # huffman_embed_rate += 2 * min(left.prob, right.prob)
        search_path = None
        if left.search_path is not None:
            search_path = -1
        elif right.search_path is not None:
            search_path = 1
        b.appendleft(Node(prob, left, right, search_path=search_path))
    root = b[0] if len(b) > 0 else a[0]
    # return root, huffman_embed_rate
    return root





def encode_step(probs, indices, message_bits, baseline):
    if baseline:
        print(f"执行baseline的版本...")

    msg_exhausted_flag = False

    sampled_index = 0
    n_bits = 0

    node = create_huffman_tree(indices, probs)
    len_message_bits = len(message_bits)

    while not node.is_leaf():  # 如果不是叶子节点，则往下执行
        probs_sum = node.prob
        ptr = random.random()
        ptr_0 = ptr * probs_sum  # 随机数乘以概率和，计算得到用于采样的r
        ptr_1 = (ptr + 0.5) * probs_sum # 随机数+0.5 再乘以概率和，也就是计算右旋后的r
        # print(f"ptr_0:{ptr_0},ptr_1:{ptr_1},node.prob:{probs_sum}")

        if ptr_1 > probs_sum:
            ptr_1 -= probs_sum
        path_table = {}

        # 记录ptr_0这个随机数对应选择的节点，分布旋转等价于随机数旋转，不同的随机数选择了不同的节点，等价于选择从不同的分布副本采样
        path_table['0'] = node.left if ptr_0 < node.left.prob else node.right
        # print(f"pathtable[0]:{path_table['0']}")
        # 记录ptr_1这个随机数对应选择的节点，分布旋转等价于随机数旋转，不同的随机数选择了不同的节点，等价于选择从不同的分布副本采样
        path_table['1'] = node.left if ptr_1 < node.left.prob else node.right
        # print(f"pathtable[1]:{path_table['1']}")


        if not msg_exhausted_flag and (len_message_bits <= n_bits):  # 如果消息用完
            # print('[*] The message is exhausted and will be padded with all zeros!')
            msg_exhausted_flag = True

        node = path_table['0'] if msg_exhausted_flag else path_table[message_bits[n_bits]]

        if path_table['0'] != path_table['1']:  # 未出现歧义，可以嵌入一个bit，这个nbits可以代表message中消息的位置，从1开始。
            n_bits += 1

    sampled_index = node.index

    return sampled_index, n_bits


import time
from scipy.stats import entropy
@torch.no_grad()
def encode_discop(model, context, message_bits, token_num_need_generated, seed,device='cuda', top_p=1.0, top_k=0, baseline_flag=False, temp=1.0):
    random.seed(seed)
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    past = None
    prev = context
    generated_ids = []
    encoded_message = []
    token_generated_num = 0
    total_entropy = 0
    stat_time = 0
    model_time = 0
    message_len = len(message_bits)

    while message_len > 0:
        if token_generated_num >= token_num_need_generated:
            # print(f"Warning: Token limit ({token_num_need_generated}) reached, but message not fully encoded.")
            break
        model_time_1 = time.time()
        probs, indices, past = get_probs_past(model, prev, past, device, top_p, top_k, temp)
        model_time_2 = time.time()
        model_time += model_time_2 - model_time_1
        stat_time_1 = time.time()
        entropy_t = entropy(probs.cpu(), base=2)
        total_entropy += entropy_t
        stat_time_2 = time.time()
        stat_time += stat_time_2 - stat_time_1

        probs = probs.tolist()
        indices = indices.tolist()

        if baseline_flag:
            sampled_index, n_bits = encode_step_baseline(indices, probs, message_bits, device)
        else:
            sampled_index, n_bits = encode_step(probs, indices, message_bits, baseline_flag)


        encoded_message.append(message_bits[:n_bits])
        generated_ids.append(sampled_index)
        message_bits = message_bits[n_bits:]
        message_len = len(message_bits)
        token_generated_num += 1
        prev = torch.tensor([sampled_index], device=device).unsqueeze(0)

    return generated_ids, encoded_message, total_entropy, stat_time, model_time



def decode_step(probs, indices, tokenID):

    node = create_huffman_tree(indices, probs, search_for=tokenID)

    path_table = [0, 0]  # 用于存储路径信息
    path_table_swap: Dict[int, str] = {}
    message_decoded_t = ''  # 初始化解码消息
    while not node.is_leaf():  # 如果不是叶子节点，则往下执行
        probs_sum = node.prob
        ptr = random.random()
        ptr_0 = ptr * probs_sum  # 随机数乘以概率和，计算得到用于采样的r

        ptr_1 = (ptr + 0.5) * probs_sum # 随机数+0.5 再乘以概率和，也就是计算右旋后的r

        if ptr_1 > probs_sum:
            ptr_1 -= probs_sum

        partition = node.left.prob

        if ptr_0 < partition:
            path_table[0] = -1
        else:
            path_table[0] = 1

        if ptr_1 < partition:
            path_table[1] = -1
        else:
            path_table[1] = 1

        if path_table[0] != path_table[1]: # 能嵌入一个bit
            if path_table[0] == -1:
                path_table_swap[-1] = '0'
                path_table_swap[1] = '1'
            else:
                path_table_swap[-1] = '1'
                path_table_swap[1] = '0'
            message_decoded_t += path_table_swap[node.search_path]

            # walk
            if node.search_path == -1:
                node = node.left
            else:
                node = node.right

        else:
            if path_table[0] == -1:
                node = node.left
            else:
                node = node.right
    return message_decoded_t


@torch.no_grad()
def decode_discop(model, context, generated_ids, seed, device='cuda', top_p=1.0, top_k=0, baseline_flag=False, temp=1.0):
    random.seed(seed)
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    past = None
    prev = context
    decoded_message = []

    for tokenID in generated_ids:
        probs, indices, past = get_probs_past(model, prev, past, device, top_p, top_k, temp)
        probs = probs.tolist()
        indices = indices.tolist()

        if baseline_flag:
            cur_decode_message = decode_step_baseline(probs, indices, tokenID, device)
        else:
            cur_decode_message = decode_step(probs, indices, tokenID)

        decoded_message.append(cur_decode_message)
        prev = torch.tensor([tokenID], device=device).unsqueeze(0)

    return decoded_message


def encode_step_baseline(indices, probs, message_bits, device):
    sampled_index = 0
    n_bits = 0

    probs_cumsum = torch.tensor(probs).cumsum(dim=0).to(device)
    interval_begin = torch.cat((torch.tensor([0],device=probs_cumsum.device), probs_cumsum[:-1]), dim=0)

    capacity = int(log2(1 / probs[0]))
    capacity_upper_bound = capacity + 1

    tbl = {} # message_bits -> token_index
    ptr = random.random()

    while capacity <= capacity_upper_bound:
        if capacity == 0:
            capacity += 1
            continue
        rotate_step_size = 2.0 ** -capacity
        is_available = True
        tbl_new = {}

        for i in range(2**capacity):
            ptr_i = ptr + i * rotate_step_size
            if ptr_i >= 1.0:
                ptr_i -= 1
            index_idx = (ptr_i >= interval_begin).nonzero()[-1].item()
            index = indices[index_idx]

            if index in tbl_new.values():
                is_available = False
                break
            tbl_new[i] = index

        if not is_available:
            break
        tbl = tbl_new
        n_bits = capacity
        capacity += 1

    if n_bits < 1:
        sampled_index = indices[(ptr >= interval_begin).nonzero()[-1].item()]
    else:
        cur_message_bits_decimal = 0
        base = 1
        for d in range(n_bits - 1, -1, -1):
            if message_bits[d] == '1':
                cur_message_bits_decimal += base
            base *= 2
        sampled_index = tbl[cur_message_bits_decimal]
    return sampled_index, n_bits


def decode_step_baseline(probs, indices, tokenID, device):
    probs_cumsum = torch.tensor(probs).cumsum(dim=0).to(device)
    interval_begin = torch.cat((torch.tensor([0],device=probs_cumsum.device), probs_cumsum[:-1]), dim=0)

    capacity = int(log2(1 / probs[0]))
    capacity_upper_bound = capacity + 1
    n_bits = 0
    message_decoded_t = ''

    tbl = {} # message_bits -> token_index
    ptr = random.random()

    while capacity <= capacity_upper_bound:
        if capacity == 0:
            capacity += 1
            continue
        rotate_step_size = 2.0 ** -capacity
        is_available = True
        tbl_new = {}

        for i in range(2**capacity):
            ptr_i = ptr + i * rotate_step_size
            if ptr_i >= 1.0:
                ptr_i -= 1

            index_idx = (ptr_i >= interval_begin).nonzero()[-1].item()
            index = indices[index_idx]

            if index in tbl_new.values():
                is_available = False
                break

            tbl_new[i] = index

        if not is_available:
            break

        tbl = tbl_new
        n_bits = capacity
        capacity += 1

    if n_bits < 1:
        message_decoded_t = ''
    else:
        if tokenID not in tbl.values(): # Error
            print(f"tokenID:{tokenID}")
            print(f"tbl.values():{tbl.values()}")
            message_decoded_t = b'x'
        else:
            tbl_swapped = {v: k for k, v in tbl.items()}  # token_index -> message bits
            message_decoded_t = bin(tbl_swapped[tokenID])[2:].zfill(n_bits)
    return message_decoded_t


