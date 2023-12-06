from Utils.Initializer import Initializer
from Utils.Utils import generate_text

# initializing
model, tokenizer, _, _, _, device, _, _, _ = Initializer()

# write you prompt here
prompt = "ismail ben alla is the next elon mask because"

# complate my text
completed_text = generate_text(model,
                               tokenizer,
                               prompt,
                               device,
                               num_samples=1,
                               max_tokens=50,
                               temp=1,
                               top_k=100)
