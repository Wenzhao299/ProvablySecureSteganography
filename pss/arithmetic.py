import torch
import torch.nn.functional as F
from utils import limit_past, get_probs_past, bits2int, int2bits, num_same_from_beg
from scipy.stats import entropy
import time

def encode_arithmetic(model, message, context, token_num_need_generation, device='cuda', precision=32, top_p=1.0, top_k=0, temp=1.0):
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2**precision
    threshold = 2**(-precision)
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    prev = context
    past = None

    with torch.no_grad():
        token_num = 0
        i = 0 # 当前处理的消息的索引
        encoded_message = []
        generated_ids = []
        total_entropy = 0
        stat_time = 0
        model_time = 0
        message_len = len(message)

        while i < message_len:
            if token_num >= token_num_need_generation:
                # print(f"Warning: Token limit ({token_num_need_generation}) reached, but message not fully encoded. Encoded {i} of {message_len} bits.")
                break
            model_time_1 = time.time()
            probs, indices, past = get_probs_past(model, prev, past, device, top_p, top_k, temp)
            past = limit_past(past)
            model_time_2 = time.time()
            model_time += model_time_2 - model_time_1

            stat_time_1 = time.time()
            entropy_t = entropy(probs.cpu(), base=2)
            total_entropy += entropy_t
            stat_time_2 = time.time()
            stat_time += stat_time_2 - stat_time_1

            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1/cur_int_range

            probs = probs.double()

            if (probs < cur_threshold).nonzero().numel() > 0:
                k = max(2, (probs < cur_threshold).nonzero()[0].item())  # not less than 2
                # probs_int = probs[:k]  # Cutoff all but top k
                probs_int = probs.clone()
                probs_int[probs < cur_threshold] = 0
            else:
                probs_int = probs.clone()


            # k = max(2, (probs < cur_threshold).nonzero()[0].item())
            # probs_int = probs[:k]

            # Rescale to correct range
            probs_int = probs_int/probs_int.sum()*cur_int_range

            # Round probabilities to integers given precision
            probs_temp_int = probs_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range - cum_probs[-1]  # add

            # Get out resulting probabilities
            probs_final = cum_probs.clone()
            probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

            # Convert to position in range
            cum_probs += cur_interval[0]

            # Get selected index based on binary fraction from message bits
            message_bits = message[i:i + precision]
            if i + precision > message_len:
                message_bits += '0' * (i + precision - message_len)
            message_list = [int(b) for b in message_bits]
            message_idx = bits2int(reversed(message_list))
            selection = (cum_probs > message_idx).nonzero()[0].item()
            # print(f"selection:{selection}")

            # Calculate new range as ints
            new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(
                reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

            # Consume most significant bits which are now fixed and update interval
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            encoded_message.append(message[i:i+num_bits_encoded])
            i += num_bits_encoded

            new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded
            new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

            cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1  # +1 here because upper bound is exclusive

            token_num += 1
            sampled_index = indices[selection].item()
            generated_ids.append(sampled_index)
            prev = torch.tensor([sampled_index], device=device).unsqueeze(0)

    return generated_ids, "".join(encoded_message), total_entropy, stat_time, model_time


def decode_arithmetic(model, generated_ids, context, device='cuda',precision=32, top_p=1.0, top_k=0, temp=1.0):

    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    i = 0

    max_val = 2**precision
    threshold = 2**(-precision)
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    prev = context
    past = None
    message = []

    with torch.no_grad():
        i = 0
        for tokenID in generated_ids:
            probs, indices, past = get_probs_past(model, prev, past, device, top_p, top_k, temp)
            past = limit_past(past)

            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1 / cur_int_range

            probs = probs.double()

            if (probs < cur_threshold).nonzero().numel() > 0:
                k = max(2, (probs < cur_threshold).nonzero()[0].item())  # not less than 2
                probs_int = probs[:k]  # Cutoff all but top k
            else:
                probs_int = probs.clone()
            # k = max(2, (probs < cur_threshold).nonzero()[0].item())
            # probs_int = probs[:k]

            # Rescale to correct range
            probs_int = probs_int / probs_int.sum() * cur_int_range

            # Round probabilities to integers given precision
            probs_temp_int = probs_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range - cum_probs[-1]  # add

            # Covnert to position in range
            cum_probs += cur_interval[0]

            rank = (indices == tokenID).nonzero().item()

            selection = rank

            # Calculate new range as ints
            new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(
                reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)

            if i == len(generated_ids) - 1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]
            cur_decoded_message = ''.join(map(str, new_bits))
            message.append(cur_decoded_message)

            new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded
            new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

            cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1  # +1 here because upper bound is exclusive

            # Update history with new token
            prev = torch.tensor([tokenID], device=device, dtype=torch.long).unsqueeze(0)
            i += 1

    return "".join(message)
