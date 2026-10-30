# Phase 3 - GenAI Libraries: Module 09 (LangChain)
# Automated Test Suite: Runs all scripts sequentially and reports results

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    ("02_chat_models.py", "9.2 Chat Models"),
    ("03_prompt_templates.py", "9.3 Prompt Templates"),
    ("04_lcel_and_routing.py", "9.4 LCEL & Routing"),
    ("05_output_parsers.py", "9.5 Output Parsers"),
    ("06_document_loaders_and_splitters.py", "9.6 Document Loaders & Splitters"),
    ("07_conversational_memory.py", "9.7 Conversational Memory"),
    ("08_full_rag_chain.py", "9.8 Full RAG Chain"),
    ("09_exercises.py", "9.9 Module 09 Exercises"),
]


def run_test_suite() -> int:
    current_dir = Path(__file__).parent
    print("=" * 70, flush=True)
    print("Running Phase 3 - Module 09 LangChain Verification Suite", flush=True)
    print("=" * 70, flush=True)

    passed = 0
    failed = 0

    for script_name, description in SCRIPTS:
        script_path = current_dir / script_name
        if not script_path.exists():
            print(f"[MISSING] {script_name} - {description}")
            failed += 1
            continue

        print(f"\n>> Executing: {script_name} ({description})...", flush=True)
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(current_dir),
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"   [PASSED] in {elapsed:.2f}s")
            passed += 1
        else:
            print(f"   [FAILED] in {elapsed:.2f}s (Exit code: {result.returncode})")
            print("--- STDERR ---")
            print(result.stderr[:400])
            failed += 1

    print("\n" + "=" * 70, flush=True)
    print(f"Summary: {passed} passed, {failed} failed out of {len(SCRIPTS)} scripts.", flush=True)
    print("=" * 70, flush=True)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_test_suite())
