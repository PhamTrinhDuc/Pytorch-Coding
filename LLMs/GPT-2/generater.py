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