import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_lora_loading():
    """Test if LoRA is being loaded correctly."""
    model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
    checkpoint_path = "GRPO/checkpoint-1000"
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    
    print("Loading LoRA adapter...")
    try:
        # Load the LoRA adapter
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        print(f"✓ LoRA loaded successfully!")
        print(f"Model type: {type(peft_model)}")
        print(f"Is PeftModel: {isinstance(peft_model, PeftModel)}")
        
        # Print trainable parameters
        print("\nTrainable parameters:")
        peft_model.print_trainable_parameters()
        
        # Check if model is in training mode (might affect LoRA application)
        print(f"\nModel training mode: {peft_model.training}")
        
        # Get some model weights to compare
        print("\nChecking if LoRA weights are loaded...")
        for name, module in peft_model.named_modules():
            if hasattr(module, 'lora_A'):
                print(f"Found LoRA adapter in: {name}")
                break
        else:
            print("❌ No LoRA adapters found in model!")
            
        # Try merging and unmerging (this verifies LoRA is working)
        print("\nTesting LoRA merge/unmerge...")
        try:
            peft_model.merge_adapter()
            print("✓ Successfully merged LoRA adapter")
            peft_model.unmerge_adapter()
            print("✓ Successfully unmerged LoRA adapter")
        except Exception as e:
            print(f"❌ Error with merge/unmerge: {e}")
            
    except Exception as e:
        print(f"❌ Error loading LoRA: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_lora_loading() 