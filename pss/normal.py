import torch
import torch.nn.functional as F
from transformers import DynamicCache

from utils import filter_logits

def encode(model, enc, context_tokens, max_tokens, top_p=1.0, top_k=0, temp=1.0, device='cuda'):
    """
    Generates text token by token using standard sampling from the full distribution.
    This serves as the "normal" baseline for comparison.
    """
    context_tensor = torch.tensor(context_tokens[-1022:], device=device, dtype=torch.long)
    
    prev = context_tensor
    output = context_tensor.clone()
    past = None

    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_tokens):
            out = model(prev.unsqueeze(0), past_key_values=DynamicCache.from_legacy_cache(past), use_cache=True)
            logits, past = out.logits, out.past_key_values

            next_token_logits = logits[0, -1, :]
            scaled_logits = next_token_logits / temp

            if top_k > 0:
                # 1. 找出第 k 大的数值 (Filter value)
                # top_k()[0] 返回值(values)，[..., -1] 取出其中最小的那个（也就是第 k 大的）
                top_k_values = torch.topk(scaled_logits, top_k)[0][..., -1, None]
                # 2. 将所有小于阈值的位置设为负无穷
                indices_to_remove = scaled_logits < top_k_values
                scaled_logits[indices_to_remove] = -float('Inf')

            if 0 < top_p < 1.0:
                scaled_logits = filter_logits(scaled_logits, top_p)
            
            probs = F.softmax(scaled_logits, dim=0)
            next_token_id = torch.multinomial(probs, num_samples=1)

            generated_tokens.append(next_token_id.item())
            
            output = torch.cat((output, next_token_id))
            prev = next_token_id

    # To maintain consistency with other encode functions, 
    # we return a tuple, although other values are dummies.
    return generated_tokens, None, None, None, None