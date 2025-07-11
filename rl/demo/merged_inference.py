import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_and_merge_model(checkpoint_path="GRPO/checkpoint-1000"):
    """
    Load the base model and merge LoRA weights permanently.
    This approach ensures the trained weights are always applied.
    """
    model_id = "HuggingFaceTB/SmolLM-135M-Instruct"    
    print(f"Loading base model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    
    print(f"Loading and merging LoRA adapter from: {checkpoint_path}")
    
    # Load the PEFT model
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    
    # Merge the LoRA weights into the base model permanently
    merged_model = peft_model.merge_and_unload()
    
    print("✓ LoRA weights merged into base model")
    print(f"Merged model type: {type(merged_model)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return merged_model, tokenizer

def generate_response_merged(model, tokenizer, prompt, max_new_tokens=50, temperature=0.01):
    """Generate response with merged model."""
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    print(f"Input length: {input_length} tokens")
    print(f"Generating max {max_new_tokens} new tokens...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    generated_tokens = outputs[0][input_length:]
    num_tokens = len(generated_tokens)
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return response, num_tokens

def compare_base_vs_merged():
    """Compare base model vs merged model to verify training effect."""
    model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load merged model
    print("\nLoading merged model...")
    merged_model, _ = load_and_merge_model()
    
    test_prompt = """SUBREDDIT: r/tifu

TITLE: TIFU by accidentally drinking soap.

POST: Ok so I was washing my hands to take my contact lenses out. As anyone with contacts with tell you, you wash with soap prior to prevent infection.

Anyway, I press down on the soap and liquid cleanliness pours onto my hands. I do my thing and dispose of the lenses. Next I fill my pink ice-age cup up with water to drink from. I lift it up and notice a strange smell, like lavender... Nevertheless, I swallow my water and realize I've made a huge mistake. A huge glob of liquid cleanliness flew into the cup and mixed with the water to become an incognito concoction of lavender and H2O. It took several swigs of mouthwash to not be a soapy-breathing-dragon.

TL;DR:
    """
    
    print(f"\n{'='*60}")
    print("COMPARISON: Base vs Merged Model")
    print(f"{'='*60}")
    print(f"Prompt: {test_prompt}")
    
    # Generate with base model
    print(f"\n{'-'*30}")
    print("BASE MODEL:")
    print(f"{'-'*30}")
    base_response, base_tokens = generate_response_merged(
        base_model, tokenizer, test_prompt, max_new_tokens=200
    )
    print(f"Response: {base_response}")
    print(f"Tokens: {base_tokens}")
    
    # Generate with merged model
    print(f"\n{'-'*30}")
    print("MERGED MODEL (with LoRA):")
    print(f"{'-'*30}")
    merged_response, merged_tokens = generate_response_merged(
        merged_model, tokenizer, test_prompt, max_new_tokens=200
    )
    print(f"Response: {merged_response}")
    print(f"Tokens: {merged_tokens}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"Base model tokens: {base_tokens}")
    print(f"Merged model tokens: {merged_tokens}")
    print(f"Token difference: {merged_tokens - base_tokens}")
    print(f"Target was 50 tokens")
    print(f"Responses are {'identical' if base_response == merged_response else 'different'}")
    print(f"{'='*60}")

def main():
    print("=== MERGED MODEL INFERENCE TEST ===")
    compare_base_vs_merged()

if __name__ == "__main__":
    main() 