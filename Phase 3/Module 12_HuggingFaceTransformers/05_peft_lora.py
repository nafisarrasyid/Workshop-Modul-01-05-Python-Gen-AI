"""Module 12 - Hugging Face Transformers
Section 12.5: PEFT & LoRA (Parameter-Efficient Fine-Tuning)

Demonstrates:
  - Configuring Low-Rank Adaptation (LoRA) via LoraConfig
  - Injecting adapter matrices into target attention layers
  - Calculating trainable vs. total parameters (demonstrating < 1% footprint)
  - Mathematical principles of LoRA decomposition (W = W0 + B * A)
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    print("=" * 60)
    print("Module 12.5: PEFT & LoRA Parameter-Efficient Fine-Tuning")
    print("=" * 60)

    try:
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType

        model_id = "hf-internal-testing/tiny-random-gpt2"
        base_model = AutoModelForCausalLM.from_pretrained(model_id)

        # 1. Define LoRA Hyperparameters
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,                  # Rank of low-rank matrices
            lora_alpha=16,        # Scaling factor (alpha / r)
            lora_dropout=0.05,    # Dropout on adapter inputs
            target_modules=["c_attn"],  # Attention projection layers
            bias="none",
        )

        # 2. Inject LoRA adapters
        peft_model = get_peft_model(base_model, peft_config)

        print("LoRA Adapter Configuration:")
        print(f"  Rank (r)            : {peft_config.r}")
        print(f"  LoRA Alpha (scaling): {peft_config.lora_alpha}")
        print(f"  Target Modules      : {peft_config.target_modules}")
        print(f"  Dropout             : {peft_config.lora_dropout}\n")

        print("Parameter Efficiency Analysis:")
        peft_model.print_trainable_parameters()

    except Exception as e:
        print(f"[Note] In-memory LoRA calculation ({e}):\n")
        # Exact parameter math demonstration
        d_model = 4096
        num_layers = 32
        r = 8
        lora_alpha = 16

        # Standard linear layer parameters: d_model * d_model
        dense_params_per_layer = d_model * d_model * 2  # q_proj and v_proj
        total_dense_params = dense_params_per_layer * num_layers

        # LoRA decomposition: W = W0 + (B * A) * (alpha / r)
        # Matrix A: d_model * r, Matrix B: r * d_model -> 2 * d_model * r
        lora_params_per_layer = (2 * d_model * r) * 2
        total_lora_params = lora_params_per_layer * num_layers

        percent = (total_lora_params / total_dense_params) * 100

        print(f"Target Model Dimension : {d_model}")
        print(f"Total Base Layers      : {num_layers}")
        print(f"LoRA Rank (r)          : {r}")
        print(f"Target Attention Params: {total_dense_params:,}")
        print(f"Trainable LoRA Params  : {total_lora_params:,}")
        print(f"Trainable Parameter %  : {percent:.3f}% (Trainable < 0.4%!)")

    print("\nMathematical Principle of LoRA:")
    print("  Given frozen weight matrix W0 in R^{d x k}:")
    print("  W = W0 + delta_W = W0 + (alpha / r) * (B @ A)")
    print("  where A in R^{r x k} is initialized with Gaussian noise and B in R^{d x r} is zero.")
    print("  This ensures delta_W = 0 at the start of training, preserving base model knowledge.")

    print("\nPEFT LoRA demonstration completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
