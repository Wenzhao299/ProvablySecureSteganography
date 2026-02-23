import torch
from utils import get_probs_past,dec2bin
from typing import List, Optional, Union
from math import log2, floor
import queue
from scipy.stats import entropy
import time


def find_nearest(anum: float, probs: List[float]) -> int:
    # Returns index_idx (index of indices)
    up = len(probs) - 1
    if up == 0:
        return 0
    bottom = 0
    while up - bottom > 1:
        index_idx = int((up + bottom) / 2)
        if probs[index_idx] < anum:
            up = index_idx
        elif probs[index_idx] > anum:
            bottom = index_idx
        else:
            return index_idx
    if up - bottom == 1:
        if probs[bottom] - anum < anum - probs[up]:
            index_idx = bottom
        else:
            index_idx = up
    return index_idx


class ADGNode:
    final_probs = []
    final_indices = []

    # node of tree
    def __init__(self,
                 probs: Union[torch.Tensor, List[float]],
                 indices: Union[torch.Tensor, List[int]],
                 multiplier: float = 1.0) -> None:
        self.probs = probs  # Guaranteed to sort in descending order of probability
        if type(self.probs) == torch.Tensor:
            self.probs = self.probs.tolist()
        self.indices = indices
        if type(self.indices) == torch.Tensor:
            self.indices = self.indices.tolist()

        self.probs_sum = sum(self.probs)
        self.children = []
        self.multiplier = multiplier
        # self.grouping()

    def grouping(self, device) -> None:
        probs = self.probs[:]
        indices = self.indices[:]

        if self.is_leaf():
            ADGNode.final_probs.extend(list(x * self.multiplier for x in self.probs))
            ADGNode.final_indices.extend(self.indices)
            return

        prob_max = max(probs)
        probs_sum = self.probs_sum
        num_groups = 2**floor(-log2(prob_max / probs_sum))
        for i in range(num_groups - 1):
            probs_child_i = []
            indices_child_i = []

            mean_probs_sum_per_group = sum(probs) / (num_groups - i)
            probs_child_i.append(probs[0])
            indices_child_i.append(indices[0])
            del probs[0]
            del indices[0]
            while True:
                delta = mean_probs_sum_per_group - sum(probs_child_i)
                if delta <= 0:
                    break
                index_idx = find_nearest(delta, probs)
                if probs[index_idx] - delta < delta:
                    probs_child_i.append(probs[index_idx])
                    indices_child_i.append(indices[index_idx])
                    del probs[index_idx]
                    del indices[index_idx]
                else:
                    break
            probs_child_i = torch.tensor(probs_child_i, device=device)
            indices_child_i = torch.tensor(indices_child_i, device=device)

            # sorting
            probs_child_i, indices_idx = probs_child_i.sort(descending=True)
            indices_child_i = indices_child_i[indices_idx]

            probs_child_i = probs_child_i.tolist()
            indices_child_i = indices_child_i.tolist()

            self.children.append(
                ADGNode(probs_child_i, indices_child_i, self.multiplier / sum(probs_child_i) * (probs_sum / num_groups)))
        # groups.append({'probs': probs, 'indices': indices})  # sorted
        self.children.append(ADGNode(probs, indices, self.multiplier / sum(probs) * (probs_sum / num_groups)))

    def is_leaf(self) -> bool:
        if max(self.probs) > self.probs_sum / 2:
            return True
        return False

    def get_final_probs_indices(self):
        # print(len(ADGNode.final_probs))
        final_probs = ADGNode.final_probs[:]
        final_indices = ADGNode.final_indices[:]
        ADGNode.final_probs = []
        ADGNode.final_indices = []
        return final_probs, final_indices



def adg_encode_step(
        probs: torch.Tensor,
        indices: torch.Tensor,
        message_bits,
        device='cuda',
        need_full_distribution=True
):

    def grouping(probs, indices):
        prob_max = probs[0]
        num_groups = 2 ** floor(-log2(prob_max))
        groups = []
        for i in range(num_groups - 1):
            mean_probs_sum_per_group = sum(probs) / (num_groups - i)
            groups.append({'probs': [], 'indices': []})
            groups[i]['probs'].append(probs[0])
            groups[i]['indices'].append(indices[0])
            del probs[0]
            del indices[0]

            while True:
                delta = mean_probs_sum_per_group - sum(groups[i]['probs'])
                if delta <= 0:
                    break
                index_idx = find_nearest(delta, probs)
                if probs[index_idx] - delta < delta:
                    groups[i]['probs'].append(probs[index_idx])
                    groups[i]['indices'].append(indices[index_idx])
                    del probs[index_idx]
                    del indices[index_idx]
                else:
                    break
        groups.append({'probs': probs, 'indices': indices})
        return groups

    total_code_len = 0

    if need_full_distribution:
        original_probs = probs.clone()
        original_indices = indices.clone()

        original_indices, indices_idx = original_indices.sort()

        original_probs = original_probs[indices_idx]

        original_probs = original_probs.cpu().numpy()

        node_q = queue.Queue()
        node_q.put(ADGNode(probs, indices))
        while not node_q.empty():
            node = node_q.get()
            node.grouping(device)
            for x in node.children:
                node_q.put(x)

        final_probs, final_indices = ADGNode.get_final_probs_indices()
        final_probs = torch.tensor(final_probs, device=device)
        final_indices = torch.tensor(final_indices, device=device)

        final_indices, indices_idx = final_indices.sort()
        final_probs = final_probs[indices_idx]

        final_probs = final_probs.cpu().numpy()

    while probs[0].item() <= 0.5:
        probs = probs.tolist()
        indices = indices.tolist()

        groups = grouping(probs,indices)
        code_len = floor(log2(len(groups)))

        bits_slice = message_bits[total_code_len:total_code_len+code_len]
        if len(bits_slice) < code_len:
            bits_slice = bits_slice.ljust(code_len, '0')
        selected_group_idx = int(bits_slice,2)
        probs = groups[selected_group_idx]['probs']
        indices = groups[selected_group_idx]['indices']

        probs, indices_idx = torch.tensor(probs, device=device).sort(descending=True)
        indices = torch.tensor(indices, device=device)

        probs = probs / probs.sum(dim=-1)
        indices = indices[indices_idx]
        total_code_len += code_len

    selected = indices[torch.multinomial(probs, 1).item()].item()

    return selected, total_code_len

# 查找 tokenid 对应的分组
def find_group_by_tokenid(groups, tokenid):
    for i, group in enumerate(groups):
        if tokenid in group['indices']:
            return i
    return None


def adg_decode_step(tokenID, probs: torch.Tensor,indices: torch.Tensor,device):
    def grouping(probs, indices):
        prob_max = probs[0]
        num_groups = 2 ** floor(-log2(prob_max))
        groups = []
        for i in range(num_groups - 1):
            mean_probs_sum_per_group = sum(probs) / (num_groups - i)
            groups.append({'probs': [], 'indices': []})
            groups[i]['probs'].append(probs[0])
            groups[i]['indices'].append(indices[0])
            del probs[0]
            del indices[0]

            while True:
                delta = mean_probs_sum_per_group - sum(groups[i]['probs'])
                if delta <= 0:
                    break
                index_idx = find_nearest(delta, probs)
                if probs[index_idx] - delta < delta:
                    groups[i]['probs'].append(probs[index_idx])
                    groups[i]['indices'].append(indices[index_idx])
                    del probs[index_idx]
                    del indices[index_idx]
                else:
                    break
        groups.append({'probs': probs, 'indices': indices})
        return groups

    cur_extracted_message = ""
    while probs[0].item() <= 0.5:
        probs = probs.tolist()
        indices = indices.tolist()

        groups = grouping(probs,indices)
        code_len = floor(log2(len(groups)))

        group_id = find_group_by_tokenid(groups, tokenID)
        if group_id is not None:
            group_id_bin = dec2bin(group_id, code_len)
            cur_extracted_message += group_id_bin
        probs = groups[group_id]['probs']
        indices = groups[group_id]['indices']

        probs, indices_idx = torch.tensor(probs, device=device).sort(descending=True)
        indices = torch.tensor(indices, device=device)

        probs = probs / probs.sum(dim=-1)
        indices = indices[indices_idx]

    return cur_extracted_message


@torch.no_grad()
def encode_adg(model, context, message_bits, token_num_need_generated,device='cuda',top_p=1.0, top_k=0, temp=1.0):
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    past = None
    prev = context
    generated_ids = []
    encoded_messages = []
    token_num_generated = 0
    total_entropy = 0
    stat_time = 0
    model_time = 0
    message_len = len(message_bits)

    while message_len > 0:
        if token_num_generated >= token_num_need_generated:
            # print(f"Warning: Token limit ({token_num_need_generated}) reached, but message not fully encoded.")
            break
        model_time_1 = time.time()
        probs, indices, past = get_probs_past(model=model,
                                              prev=prev,
                                              past=past,
                                              device=device,
                                              top_p=top_p,
                                              top_k=top_k,
                                              temp=temp)
        model_time_2 = time.time()
        model_time += model_time_2 - model_time_1

        stat_time_1 = time.time()
        entropy_t = entropy(probs.cpu(),base=2)
        total_entropy += entropy_t
        stat_time_2 = time.time()
        stat_time += stat_time_2 - stat_time_1

        sampled_index, total_code_len = adg_encode_step(probs=probs,
                                                        indices=indices,
                                                        message_bits=message_bits,
                                                        device=device,
                                                        need_full_distribution=False)
        cur_encoded_message = message_bits[:total_code_len]
        message_bits = message_bits[total_code_len:]
        message_len = len(message_bits)
        encoded_messages.append(cur_encoded_message)

        indices = indices.tolist()
        indices_idx = indices.index(sampled_index)
        generated_ids.append(sampled_index)
        prev = torch.tensor([sampled_index], device=device, dtype=torch.long).unsqueeze(0)
        token_num_generated += 1
    return generated_ids, encoded_messages, total_entropy, stat_time, model_time



@torch.no_grad()
def decode_adg(model, generated_ids, context, device='cuda',top_p=1.0, top_k=0, temp=1.0):
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    past = None
    prev = context
    extracted_message = []

    for tokenID in generated_ids:
        probs, indices, past = get_probs_past(model=model,
                                              prev=prev,
                                              past=past,
                                              device=device,
                                              top_p=top_p,
                                              top_k=top_k,
                                              temp=temp)
        cur_extracted_message = adg_decode_step(probs=probs,
                                                      indices=indices,
                                                      tokenID=tokenID,
                                                       device=device)
        extracted_message.append(cur_extracted_message)

        prev = torch.tensor([tokenID], device=device, dtype=torch.long).unsqueeze(0)

    return extracted_message


