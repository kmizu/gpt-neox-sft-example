import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

prompt = [
    { "speaker": "ユーザー", "text": "日本について教えてください" }
]
prompt = [
    f"{uttr['speaker']}: {uttr['text']}" for uttr in prompt
]
prompt = "<NL>".join(prompt)
prompt = (
    prompt + "<NL>" + "システム: "
)

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft")

if torch.cuda.is_available():
    model = model.to('cuda')

token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        do_sample=True,
        max_new_tokens=1024,
        temperature=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
output = output.replace("<NL>", "\n")
print(output)
