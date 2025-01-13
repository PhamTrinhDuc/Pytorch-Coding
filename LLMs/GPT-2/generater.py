import torch
import tiktoken
from model import GPTModel, GPTConfig124M

def text_to_tokens_ids(text: str, tokenizer) -> torch.Tensor:
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long) 
    encoded_tensor = token_ids.unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(tokens_ids: torch.Tensor, tokenizer) -> str:
    flat = tokens_ids.squeeze(0) # remove batch dimension
    text = tokenizer.decode(flat.tolist())
    return text


def top_k_sampling(logits: torch.Tensor, top_k: int):
    # Lấy top_k token có xác suất cao nhất
    top_logits, top_indices = torch.topk(input=logits, k=top_k)
    
    # Chuẩn hóa lại xác suất (softmax) trên top_k logits
    probabilities = torch.softmax(top_logits, dim=-1)
    
    # Chọn ngẫu nhiên một token từ top_k
    chosen_index = torch.multinomial(probabilities, num_samples=1)
    
    # Trả về token đã chọn
    return top_indices[chosen_index].item()


def top_p_sampling(logits: torch.Tensor, top_p: int):
    # Sắp xếp các token theo xác suất giảm dần
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    
    # Tính tổng tích lũy xác suất
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    
    # Lấy token đầu tiên sao cho tổng xác suất >= top_p
    cutoff_index = (cumulative_probabilities > top_p).nonzero(as_tuple=True)[0][0]
    
    # Chỉ giữ lại các token trong nhóm hạt nhân
    top_indices = sorted_indices[:cutoff_index + 1]
    top_logits = sorted_logits[:cutoff_index + 1]
    
    # Chuẩn hóa xác suất trên nhóm token được chọn
    probabilities = torch.softmax(top_logits, dim=-1)
    
    # Chọn ngẫu nhiên một token
    chosen_index = torch.multinomial(probabilities, num_samples=1)
    
    return top_indices[chosen_index].item()

def temperature_sampling(logits, temperature):
    # Chia logits cho temperature để điều chỉnh phân phối
    adjusted_logits = logits / temperature
    
    # Tính xác suất softmax
    probabilities = torch.softmax(adjusted_logits, dim=-1)
    
    # Chọn ngẫu nhiên một token dựa trên phân phối
    chosen_index = torch.multinomial(probabilities, num_samples=1)
    
    return chosen_index.item()

def beam_search(logits_fn, beam_width: int, max_length: int):
    beams = [([], 0)]  # (sequence, score)
    
    for _ in range(max_length):
        new_beams = []
        for seq, score in beams:
            # Tính logits cho chuỗi hiện tại
            logits = logits_fn(seq)
            probabilities = torch.softmax(logits, dim=-1)
            
            # Lấy top-k (beam_width) token
            top_logits, top_indices = torch.topk(probabilities, k=beam_width)
            
            # Tạo các beam mới
            for i in range(beam_width):
                new_seq = seq + [top_indices[i].item()]
                new_score = score + torch.log(top_logits[i]).item()
                new_beams.append((new_seq, new_score))
        
        # Chỉ giữ lại các beam tốt nhất
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        beams = new_beams
    
    # Trả về beam tốt nhất
    return beams[0][0]


def generate(model: GPTModel, 
             input: list[int], 
             max_new_tokens: int,
             context_length: int,
             temperature: int,
             top_k=None,
             eos_id=None):
    for _ in range(max_new_tokens):

        input_model = input[:, -context_length:]
        with torch.no_grad():
            logits = model(input_model) # [N, context_length, vocab_size]
        logits = logits[:, -1, :] # [N, vocab_size] # lấy từ mới được predict ra 

        if top_k is not None:
            top_logits, indices = torch.topk(input=logits, k = top_k) # [N, top_k], [N, top_k_indices]
            min_val = top_logits[:, -1] # [N, 1]
            # if logits < min_val, replace = "-inf", else return logits
            logits = torch.where(logits < min_val, 
                                 torch.tensor(float("-inf")).to(logits.device),
                                 logits) # [N, vocab_size]
        if temperature > 0.0:
            logits = logits / temperature # [N, vocab_size]

            probs = torch.softmax(input=logits, dim=-1) # [N, vocab_size]
            # choose random from top_k
            idx_next = torch.multinomial(input=probs, num_samples=1) # [N, 1]
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        if idx_next == eos_id:
            break

        input = torch.cat([input, idx_next], dim=1) # [N, context_length+1]
    
    return input

def main():
    args = GPTConfig124M
    model = GPTModel(args=args)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    INPUT_PROMPT = "Every effort moves you"
    input_ids = text_to_tokens_ids(text=INPUT_PROMPT, tokenizer=tokenizer)
    ids_generated = generate(model=model, 
                              input=input_ids, 
                              max_new_tokens=args.max_new_tokens, 
                              context_length=args.context_length, 
                              temperature=0.5,
                              top_k=3)
    text_generated = token_ids_to_text(tokens_ids=ids_generated, tokenizer=tokenizer)
    print(text_generated)
    
if __name__ == "__main__":
    main()