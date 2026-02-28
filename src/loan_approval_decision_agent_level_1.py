# Mount Google Drive (if using Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except:
    IN_COLAB = False
    print("Not running in Colab")

!pip install bitsandbytes accelerate

# -------------------------
# 0) Setup & Imports
# -------------------------
import json
import re
import torch
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


# -------------------------
# 1) Load Model (same approach)
# -------------------------
# Change these paths to match your environment
model_path = "/content/drive/MyDrive/Phi_3_5_mini_instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    local_files_only=True
)

print("✅ Model & tokenizer loaded")

df=pd.read_csv("loan_rules.csv")
print(df.head())
print("Columns:", df.columns)

# ─────────────────────────────────────────────
# STEP 1: PARSE RULES FROM YOUR DATAFRAME
# ─────────────────────────────────────────────
def parse_condition(condition_str: str) -> tuple:
    """
    Parse "amount <= 1000" → ("amount", "<=", 1000.0)
    Supports: <=, >=, <, >, =, !=
    """
    pattern = r"(\w+)\s*(<=|>=|!=|<|>|=)\s*([\d.]+)"
    match = re.match(pattern, condition_str.strip())
    if not match:
        raise ValueError(f"Cannot parse condition: '{condition_str}'")
    return (match.group(1), match.group(2), float(match.group(3)))


def load_rules_from_df(rules_df: pd.DataFrame) -> list[dict]:
    """
    Convert each DataFrame row into a rule dict.
    Row order = rule priority (first match wins).

    Required columns: section, rule_description, condition, decision, risk_level
    """
    rules = []
    for idx, row in rules_df.iterrows():
        rules.append({
            "id":            idx,
            "section":       row["section"],
            "label":         row["rule_description"],
            "condition_str": row["condition"],
            "condition":     parse_condition(row["condition"]),
            "decision":      row["decision"],
            "risk_level":    row["risk_level"],
        })
    return rules



# ─────────────────────────────────────────────
# STEP 2: EXTRACT LOAN AMOUNT FROM USER INPUT
# ─────────────────────────────────────────────
def extract_amount(user_input: str) -> float | None:
    """
    Extract first numeric value from free-text input.
    Handles: $15,000  |  15000  |  15.5k  |  "$500"
    """
    cleaned = user_input.replace(",", "")

    # Handle shorthand: "15k" → 15000
    k_match = re.search(r"\$?(\d+(?:\.\d+)?)\s*k\b", cleaned, re.IGNORECASE)
    if k_match:
        return float(k_match.group(1)) * 1000

    match = re.search(r"\$?(\d+(?:\.\d+)?)", cleaned)
    return float(match.group(1)) if match else None

def evaluate_condition(amount: float, condition: tuple) -> bool:
    """
    Manually evaluate (field, operator, threshold).
    ⚠️ Never uses eval().
    """
    _, operator, threshold = condition
    operator = operator.strip()

    if operator == "<=":
        return amount <= threshold
    elif operator == "<":
        return amount < threshold
    elif operator == ">=":
        return amount >= threshold
    elif operator == ">":
        return amount > threshold
    elif operator == "=":
        return amount == threshold
    elif operator == "!=":
        return amount != threshold
    else:
        raise ValueError(f"Unsupported operator: {operator}")

# ─────────────────────────────────────────────
# STEP 4: RULE SELECTION (first-match wins)
# ─────────────────────────────────────────────
def select_rule(amount: float, rules: list[dict]) -> dict | None:
    """
    Loop through rules in DataFrame order.
    Return FIRST rule whose condition evaluates to True.
    ⚠️ Python decides — NOT the LLM.
    """
    for rule in rules:
        if evaluate_condition(amount, rule["condition"]):
            return rule
    return None

# ─────────────────────────────────────────────
# STEP 5: LLM EXPLANATION LAYER
# ─────────────────────────────────────────────
def build_prompt(user_input: str, amount: float, rule: dict) -> str:
    return (
        "You are a professional loan officer assistant. "
        "A loan decision has already been made by our rule engine. "
        "Your ONLY job is to write a clear, professional one sentence explanation "
        "for the applicant. Do NOT change, question, or override the decision.\n\n"
        f"Applicant Request : \"{user_input}\"\n"
        f"Loan Amount       : ${amount:,.0f}\n"
        f"Loan Category     : {rule['section']}\n"
        f"Applied Rule      : {rule['label']}\n"
        f"Condition Met     : {rule['condition_str']}\n"
        f"Decision          : {rule['decision']}\n"
        f"Risk Level        : {rule['risk_level']}\n\n"
        "Explanation:"
    )


def get_llm_explanation(pipe, user_input: str, amount: float, rule: dict) -> str:
    """Call Phi-3.5-mini-instruct to generate an explanation (never the decision)."""
    messages = [{"role": "user", "content": build_prompt(user_input, amount, rule)}]
    output = pipe(messages, max_new_tokens=150, do_sample=False)

    response = output[0]["generated_text"]
    if isinstance(response, list):
        for msg in reversed(response):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg["content"].strip()
    return str(response).strip()



# ─────────────────────────────────────────────
# STEP 6: MAIN DECISION PIPELINE
# ─────────────────────────────────────────────
def process_loan_application(pipe, user_input: str, rules: list[dict]) -> dict:
    """
    Single application pipeline:
      Extract amount → Select rule (Python) → LLM explains → Structured output
    """
    amount = extract_amount(user_input)

    if amount is None:
        return {
            "user_input":    user_input,
            "amount":        None,
            "section":       "N/A",
            "matched_rule":  "N/A",
            "condition_met": "N/A",
            "decision":      "ERROR",
            "risk_level":    "N/A",
            "reasoning":     "Could not extract a loan amount from the input.",
        }

    rule = select_rule(amount, rules)

    if rule is None:
        return {
            "user_input":    user_input,
            "amount":        amount,
            "section":       "N/A",
            "matched_rule":  "No rule matched",
            "condition_met": "N/A",
            "decision":      "REJECTED",
            "risk_level":    "High",
            "reasoning":     "No applicable rule found for this loan amount.",
        }

    return {
        "user_input":    user_input,
        "amount":        amount,
        "section":       rule["section"],
        "matched_rule":  rule["label"],
        "condition_met": rule["condition_str"],
        "decision":      rule["decision"],      # ← Set by Python rule engine
        "risk_level":    rule["risk_level"],    # ← Set by Python rule engine
        "reasoning":     get_llm_explanation(pipe, user_input, amount, rule),  # ← LLM only
    }


def print_result(result: dict):
    icons = {"APPROVED": "✅", "MANUAL_REVIEW": "🔍", "REJECTED": "❌", "ERROR": "⚠️"}
    icon  = icons.get(result["decision"], "")
    amt   = f"${result['amount']:,.0f}" if result["amount"] is not None else "N/A"
    print("=" * 65)
    print(f"  Input      : {result['user_input']}")
    print(f"  Amount     : {amt}")
    print(f"  Section    : {result['section']}")
    print(f"  Rule       : {result['matched_rule']}")
    print(f"  Condition  : {result['condition_met']}")
    print(f"  Decision   : {icon} {result['decision']}")
    print(f"  Risk Level : {result['risk_level']}")
    print(f"  Reasoning  :")
    print(f"    {result['reasoning']}")
    print("=" * 65 + "\n")

# ─────────────────────────────────────────────
# STEP 7: USER QUERY FUNCTION
# ─────────────────────────────────────────────

# Load rules and model once at module level
_rules = load_rules_from_df(df)

# Initialize the text generation pipeline
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=150 # Set max_length here to match max_new_tokens in get_llm_explanation
)

def user_query(query: str):
    """
    Process a single loan request query.
    """
    result = process_loan_application(text_generation_pipeline, query, _rules)
    print_result(result)

user_query("i need 85,000 dollars") #1

user_query("I need $800 to fix my phone.") #2

user_query("I'd like to borrow $7500 for home appliances.") #3

user_query("Please approve a loan of $45,000 for renovation.") #4

user_query("Apply for 500k loan for my company.") #5

"""
### How Rule Matching Works
The rule engine reads every rule directly from your DataFrame, preserving the exact row order. When a user submits a query, the system first extracts the numeric loan amount from the free-text input using regex. It then walks through the rules one by one, from row 0 downward, and evaluates each condition — such as amount <= 1000 or amount <= 25000 — using a manual comparison function that explicitly checks the operator without ever calling Python's eval(). The moment a condition returns True, that rule is selected and the loop stops immediately. This "first-match wins" strategy means the most specific, lowest-threshold rules at the top of the DataFrame always take priority over broader rules further down. If no rule matches at all, the system defaults to REJECTED. The result is a deterministic, fully traceable decision where every outcome can be explained by pointing to a single row in your DataFrame.
### Why the LLM Does Not Decide
The LLM's role is strictly limited to writing the explanation after the decision has already been made by the rule engine. By the time the model receives any input, the decision, risk level, matched rule, and condition are all finalized and passed in as fixed context. The prompt explicitly instructs the model not to change, question, or override the decision — it is only asked to translate the structured output into a professional, human-readable sentence or two. This separation exists for two critical reasons. First, LLMs are probabilistic — given the same input twice, they may produce different outputs, which makes them fundamentally unsuitable for consistent, auditable financial decisions. Second, regulatory and compliance frameworks in lending require that every decision be fully explainable and traceable back to a documented rule, something a neural network cannot provide on its own. Keeping the LLM in an explanation-only role gives you the best of both worlds: reliable, rule-based decisions that can be audited, and natural language output that communicates those decisions clearly to the applicant.
"""
