import functools
import time


# ==========================================
# 1. token_cost
# ==========================================
def token_cost(tokens: int, model: str) -> float:
    """Returns the estimated cost using a dict of costs per 1K tokens.
    
    Raises ValueError if the model is unknown.
    """
    cost_per_1k = {
        "gpt-4o": 0.005,
        "claude-sonnet-4-5": 0.003,
        "gemini-1.5-pro": 0.00125,
    }

    if model not in cost_per_1k:
        raise ValueError(f"Unknown model: '{model}'. Available models: {list(cost_per_1k.keys())}")

    return (tokens / 1000) * cost_per_1k[model]


# ==========================================
# 2. retry decorator
# ==========================================
def retry(n: int = 3, sleep_seconds: float = 0.1):
    """Decorator that retries a function up to n times on any exception."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, n + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt} failed: {e}")
                    if attempt < n:
                        time.sleep(sleep_seconds)
            raise last_exception
        return wrapper
    return decorator


# ==========================================
# 3. temperature_label
# ==========================================
def temperature_label(t: float) -> str:
    """Maps temperature ranges to descriptive labels.
    
    0.0–0.3 -> "precise"
    0.3–0.7 -> "balanced"
    0.7–1.0 -> "creative"
    Raises ValueError outside 0.0–1.0.
    """
    if t < 0.0 or t > 1.0:
        raise ValueError(f"Temperature {t} is out of bounds (must be between 0.0 and 1.0).")

    if t <= 0.3:
        return "precise"
    elif t <= 0.7:
        return "balanced"
    else:
        return "creative"


# ==========================================
# 4. Parse string without regex
# ==========================================
raw_text = "128000 tokens, 0.005 USD per 1K"

# Example parsing: split by comma, then take the first item of each
part1, part2 = raw_text.split(",")
token_count = int(part1.strip().split()[0])
cost = float(part2.strip().split()[0])


# ==========================================
# Testing all 4 exercises
# ==========================================
if __name__ == "__main__":
    print("--- Exercise 1: token_cost ---")
    cost_val = token_cost(10000, "gpt-4o")
    print(f"Cost for 10,000 tokens (gpt-4o): ${cost_val:.4f}")
    try:
        token_cost(1000, "unknown-model")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\n--- Exercise 2: retry decorator ---")
    global call_count
    call_count = 0

    @retry(n=3, sleep_seconds=0.05)
    def flaky_function():
        global call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError(f"Simulated failure #{call_count}")
        return "Success on attempt 3!"

    print(flaky_function())

    print("\n--- Exercise 3: temperature_label ---")
    print("0.2 ->", temperature_label(0.2))
    print("0.5 ->", temperature_label(0.5))
    print("0.8 ->", temperature_label(0.8))
    try:
        temperature_label(1.5)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\n--- Exercise 4: parse string ---")
    print(f"Parsed tokens: {token_count} (type: {type(token_count).__name__})")
    print(f"Parsed cost: {cost} (type: {type(cost).__name__})")
