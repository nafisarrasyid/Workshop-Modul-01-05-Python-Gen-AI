"""Module 10 - LlamaIndex
Section 10.2: Setup & Global Settings

Demonstrates:
  - Configuring global Settings (LLM, Embedding, chunk_size, chunk_overlap)
  - Integrating environment variables and model resolution
  - Verifying settings before index creation
"""
import sys
from pathlib import Path

# Add parent directory to path to import shared utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import configure_llama_index

from llama_index.core import Settings

def main():
    print("=" * 60)
    print("Module 10.2: LlamaIndex Global Settings Configuration")
    print("=" * 60)

    # 1. Configure settings using shared robust resolver
    configure_llama_index(model="gpt-4o")

    # 2. Configure Chunking hyperparameters
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    # 3. Inspect active settings
    print(f"LLM Class           : {Settings.llm.__class__.__name__}")
    if hasattr(Settings.llm, "model"):
        print(f"LLM Model           : {Settings.llm.model}")
    print(f"Embedding Class     : {Settings.embed_model.__class__.__name__}")
    if hasattr(Settings.embed_model, "model_name"):
        print(f"Embedding Model     : {Settings.embed_model.model_name}")
    print(f"Chunk Size          : {Settings.chunk_size} tokens")
    print(f"Chunk Overlap       : {Settings.chunk_overlap} tokens")

    print("\nSettings verified successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
