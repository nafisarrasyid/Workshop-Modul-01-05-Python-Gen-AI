"""Master Test Runner for Phase 3: GenAI Libraries
Executes and validates all scripts across Modules 09 to 14.
"""
import sys
import os
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODULE_SCRIPTS = {
    "Module 09 - LangChain": [
        "Module 09_LangChain/02_chat_models.py",
        "Module 09_LangChain/03_prompt_templates.py",
        "Module 09_LangChain/04_lcel_and_routing.py",
        "Module 09_LangChain/05_output_parsers.py",
        "Module 09_LangChain/06_document_loaders_and_splitters.py",
        "Module 09_LangChain/07_conversational_memory.py",
        "Module 09_LangChain/08_full_rag_chain.py",
        "Module 09_LangChain/09_exercises.py",
    ],
    "Module 10 - LlamaIndex": [
        "Module 10_LlamaIndex/02_setup_and_settings.py",
        "Module 10_LlamaIndex/03_documents_and_nodes.py",
        "Module 10_LlamaIndex/04_vector_store_index.py",
        "Module 10_LlamaIndex/05_retriever.py",
        "Module 10_LlamaIndex/06_summary_index.py",
        "Module 10_LlamaIndex/07_persistent_storage.py",
        "Module 10_LlamaIndex/08_custom_query_pipeline.py",
        "Module 10_LlamaIndex/09_exercises.py",
    ],
    "Module 11 - Vector Databases": [
        "Module 11_VectorDatabases/02_chromadb.py",
        "Module 11_VectorDatabases/03_pinecone.py",
        "Module 11_VectorDatabases/04_weaviate.py",
        "Module 11_VectorDatabases/05_exercises.py",
    ],
    "Module 12 - Hugging Face Transformers": [
        "Module 12_HuggingFaceTransformers/02_pipeline_api.py",
        "Module 12_HuggingFaceTransformers/03_tokenizers.py",
        "Module 12_HuggingFaceTransformers/04_automodel.py",
        "Module 12_HuggingFaceTransformers/05_peft_lora.py",
        "Module 12_HuggingFaceTransformers/06_exercises.py",
    ],
    "Module 13 - LangGraph & Agents": [
        "Module 13_LangGraph/02_core_concepts.py",
        "Module 13_LangGraph/03_react_agent.py",
        "Module 13_LangGraph/04_human_in_the_loop.py",
        "Module 13_LangGraph/05_exercises.py",
    ],
    "Module 14 - CrewAI": [
        "Module 14_CrewAI/02_agents.py",
        "Module 14_CrewAI/03_tools.py",
        "Module 14_CrewAI/04_tasks.py",
        "Module 14_CrewAI/05_crew_sequential.py",
        "Module 14_CrewAI/06_hierarchical_manager.py",
        "Module 14_CrewAI/07_exercises.py",
    ],
}

def run_script(script_rel_path: str):
    full_path = BASE_DIR / script_rel_path
    if not full_path.exists():
        return False, f"File not found: {full_path}", 0.0

    t0 = time.perf_counter()
    res = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=str(full_path.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - t0
    success = (res.returncode == 0)
    err = res.stderr if not success else ""
    return success, err, elapsed

def main():
    print("=" * 80)
    print("PHASE 3: GENAI LIBRARIES - MASTER VALIDATION RUNNER")
    print("Python Interpreter:", sys.executable)
    print("=" * 80)

    total_passed = 0
    total_failed = 0
    failures = []

    for module_name, scripts in MODULE_SCRIPTS.items():
        print(f"\n[{module_name}] ({len(scripts)} scripts)")
        for script in scripts:
            script_name = Path(script).name
            success, err, elapsed = run_script(script)
            if success:
                print(f"  [PASS] {script_name:<40} ({elapsed:.2f}s)")
                total_passed += 1
            else:
                print(f"  [FAIL] {script_name:<40} ({elapsed:.2f}s)")
                total_failed += 1
                failures.append((script, err))

    print("\n" + "=" * 80)
    print(f"FINAL RESULT: {total_passed} PASSED, {total_failed} FAILED out of {total_passed + total_failed} scripts.")
    print("=" * 80)

    if failures:
        print("\nFailures Detail:")
        for script, err in failures:
            print(f"\n--- {script} ---")
            print(err[:500])
        sys.exit(1)
    else:
        print("\nAll Phase 3 scripts executed flawlessly with 0 errors!")
        sys.exit(0)

if __name__ == "__main__":
    main()
