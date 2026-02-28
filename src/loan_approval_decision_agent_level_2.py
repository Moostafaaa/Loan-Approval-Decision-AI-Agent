# Mount Google Drive (if using Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except:
    IN_COLAB = False
    print("Not running in Colab")


# -------------------------
# Setup & Imports
# -------------------------
import json
import re
import torch
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
# !pip install bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


# -------------------------
# 0) Load Model (same approach)
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


df = pd.DataFrame({
    "section": ["Loan"] * 7,
    "rule_description": [
        "Small loan auto approved",
        "Medium loan with good credit approved",
        "Medium loan review",
        "High loan with good credit manual review",
        "High loan with low credit rejected",
        "Very low credit score rejection",
        "Very large loan rejection",
    ],
    "condition": [
        "amount <= 5000",
        "5000 < amount <= 20000 AND credit_score >= 700",
        "5000 < amount <= 20000",
        "amount > 20000 AND credit_score >= 700",
        "amount > 20000 AND credit_score < 700",
        "credit_score < 600",
        "amount > 100000",
    ],
    "decision": [
        "APPROVED",
        "APPROVED",
        "MANUAL_REVIEW",
        "MANUAL_REVIEW",
        "REJECTED",
        "REJECTED",
        "REJECTED",
    ],
    "risk_level": [
        "Low",
        "Low",
        "Medium",
        "Medium",
        "High",
        "High",
        "High",
    ],
})
df.head()

# ─────────────────────────────────────────────
# STEP 1: PARSE RULES FROM YOUR DATAFRAME
# ─────────────────────────────────────────────
CHAINED_PATTERN = re.compile(
    r"([\d.]+)\s*(<=|<|>=|>)\s*(\w+)\s*(<=|<|>=|>)\s*([\d.]+)"
)
SIMPLE_PATTERN = re.compile(
    r"(\w+)\s*(<=|>=|!=|<|>|=)\s*([\d.]+)"
)


def parse_single_condition(cond_str: str) -> dict:
    """
    Parse one condition string (no AND) into a structured dict.

    Handles two forms:
      Simple  : "amount <= 5000"          → {type: simple,  field, op, threshold}
      Chained : "5000 < amount <= 20000"  → {type: chained, field, low, low_op, high, high_op}
    """
    cond_str = cond_str.strip()

    # Try chained first: e.g. "5000 < amount <= 20000"
    m = CHAINED_PATTERN.match(cond_str)
    if m:
        return {
            "type":    "chained",
            "field":   m.group(3),
            "low":     float(m.group(1)),
            "low_op":  m.group(2),
            "high":    float(m.group(5)),
            "high_op": m.group(4),
        }

    # Try simple: e.g. "amount <= 5000"
    m = SIMPLE_PATTERN.match(cond_str)
    if m:
        return {
            "type":      "simple",
            "field":     m.group(1),
            "op":        m.group(2),
            "threshold": float(m.group(3)),
        }

    raise ValueError(f"Cannot parse condition part: '{cond_str}'")


def parse_condition(condition_str: str) -> list[dict]:
    """
    Split condition on AND, parse each part.
    Returns a list of condition dicts — ALL must be True for the rule to match.
    """
    parts = [p.strip() for p in re.split(r"\bAND\b", condition_str, flags=re.IGNORECASE)]
    return [parse_single_condition(p) for p in parts]


def load_rules_from_df(rules_df: pd.DataFrame) -> list[dict]:
    """
    Convert each DataFrame row into a rule dict.
    Row order is preserved; priority is applied later in select_rule().

    Required columns: section, rule_description, condition, decision, risk_level
    """
    rules = []
    for idx, row in rules_df.iterrows():
        rules.append({
            "id":            idx,
            "section":       row["section"],
            "label":         row["rule_description"],
            "condition_str": row["condition"],
            "conditions":    parse_condition(row["condition"]),
            "decision":      row["decision"],
            "risk_level":    row["risk_level"],
        })
    return rules

# ─────────────────────────────────────────────
# STEP 2: EXTRACT MULTIPLE VARIABLES
# ─────────────────────────────────────────────
def extract_variables(user_input: str) -> dict:
    """
    Extract amount and credit_score from free-text input.

    Strategy:
      - Scan all numbers and classify by surrounding keyword context (40 chars before)
      - "credit / score / fico" before number → credit_score
      - "loan / borrow / need / want / $"  before number → amount
      - Fallback by value range: 300–850 → credit_score, otherwise → amount
    """
    cleaned   = user_input.replace(",", "")
    variables = {}

    num_pattern    = re.compile(r"\$?(\d+(?:\.\d+)?)\s*(k?)\b")
    credit_keywords = re.compile(r"credit|score|fico", re.IGNORECASE)
    loan_keywords   = re.compile(r"loan|borrow|need|want|get|apply|request|\$", re.IGNORECASE)

    for m in num_pattern.finditer(cleaned):
        pos = m.start()
        val = float(m.group(1)) * (1000 if m.group(2).lower() == "k" else 1)
        context_before = cleaned[max(0, pos - 40): pos]

        if credit_keywords.search(context_before):
            variables["credit_score"] = val
        elif loan_keywords.search(context_before):
            if "amount" not in variables:
                variables["amount"] = val
        elif 300 <= val <= 850:
            if "credit_score" not in variables:
                variables["credit_score"] = val
        else:
            if "amount" not in variables:
                variables["amount"] = val

    return variables

# ─────────────────────────────────────────────
# STEP 3: SAFE CONDITION EVALUATOR (no eval())
# ─────────────────────────────────────────────
def _apply_op(left: float, op: str, right: float) -> bool:
    """Apply a single comparison operator. No eval()."""
    ops = {
        "<=": left <= right,
        "<":  left <  right,
        ">=": left >= right,
        ">":  left >  right,
        "=":  left == right,
        "!=": left != right,
    }
    if op not in ops:
        raise ValueError(f"Unsupported operator: {op}")
    return ops[op]


def evaluate_single_condition(cond: dict, variables: dict) -> bool | None:
    """
    Evaluate one parsed condition dict against extracted variables.
    Returns:
      True / False → evaluated successfully
      None         → required variable is missing (triggers fallback)
    """
    field = cond["field"]
    if field not in variables:
        return None  # missing variable → can't evaluate

    value = variables[field]

    if cond["type"] == "simple":
        return _apply_op(value, cond["op"], cond["threshold"])

    if cond["type"] == "chained":
        # e.g. 5000 < amount <= 20000
        left_ok  = _apply_op(cond["low"],  cond["low_op"],  value)
        right_ok = _apply_op(value, cond["high_op"], cond["high"])
        return left_ok and right_ok

    raise ValueError(f"Unknown condition type: {cond['type']}")


def evaluate_rule(rule: dict, variables: dict) -> bool | None:
    """
    Evaluate ALL conditions for a rule (AND logic).
    Returns:
      True  → all conditions pass
      False → at least one condition fails
      None  → a required variable is missing
    """
    for cond in rule["conditions"]:
        result = evaluate_single_condition(cond, variables)
        if result is None:
            return None   # missing variable
        if not result:
            return False  # short-circuit AND
    return True

# ─────────────────────────────────────────────
# STEP 4: PRIORITY STRATEGY
# ─────────────────────────────────────────────
# Strategy:
#   1. Collect ALL matching rules
#   2. Pick the rule with the HIGHEST risk level (High > Medium > Low)
#   3. Tie-break: first matching rule in DataFrame order wins

RISK_PRIORITY = {"High": 3, "Medium": 2, "Low": 1}


def select_rule(variables: dict, rules: list[dict]) -> tuple[dict | None, bool]:
    """
    Returns (best_matching_rule, needs_more_info).
    needs_more_info=True means at least one rule was skipped due to a missing variable.
    """
    matched    = []
    needs_info = False

    for rule in rules:
        result = evaluate_rule(rule, variables)
        if result is True:
            matched.append(rule)
        elif result is None:
            needs_info = True

    if not matched:
        return None, needs_info

    # Stable sort: highest risk first; ties keep original DataFrame order
    matched.sort(key=lambda r: RISK_PRIORITY.get(r["risk_level"], 0), reverse=True)
    return matched[0], False

# ─────────────────────────────────────────────
# STEP 5: FALLBACK HANDLING
# ─────────────────────────────────────────────
FALLBACK_REASONING = (
    "We were unable to reach a decision because required information is missing or unclear. "
    "Please provide both your requested loan amount and your credit score."
)


def build_fallback(user_input: str, variables: dict) -> dict:
    return {
        "user_input":    user_input,
        "amount":        variables.get("amount"),
        "credit_score":  variables.get("credit_score"),
        "section":       "N/A",
        "matched_rule":  "N/A",
        "condition_met": "N/A",
        "decision":      "NEED_MORE_INFORMATION",
        "risk_level":    "Unknown",
        "reasoning":     FALLBACK_REASONING,
    }

# ─────────────────────────────────────────────
# STEP 6: LLM EXPLANATION LAYER
# ─────────────────────────────────────────────
def build_prompt(user_input: str, variables: dict, rule: dict) -> str:
    amt    = f"${variables['amount']:,.0f}" if "amount"       in variables else "N/A"
    credit = int(variables["credit_score"]) if "credit_score" in variables else "N/A"
    return (
        "You are a professional loan officer assistant. "
        "A loan decision has already been made by our rule engine. "
        "Your ONLY job is to write a clear, professional one sentence explanation "
        "for the applicant. Do NOT change, question, or override the decision.\n\n"
        f"Applicant Request : \"{user_input}\"\n"
        f"Loan Amount       : {amt}\n"
        f"Credit Score      : {credit}\n"
        f"Loan Category     : {rule['section']}\n"
        f"Applied Rule      : {rule['label']}\n"
        f"Condition Met     : {rule['condition_str']}\n"
        f"Decision          : {rule['decision']}\n"
        f"Risk Level        : {rule['risk_level']}\n\n"
        "Explanation:"
    )


def get_llm_explanation(pipe, user_input: str, variables: dict, rule: dict) -> str:
    """Call Phi-3.5-mini-instruct for explanation only — never the decision."""
    messages = [{"role": "user", "content": build_prompt(user_input, variables, rule)}]
    output   = pipe(messages, max_new_tokens=150, do_sample=False)

    response = output[0]["generated_text"]
    if isinstance(response, list):
        for msg in reversed(response):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg["content"].strip()
    return str(response).strip()

# ─────────────────────────────────────────────
# STEP 7: MAIN DECISION PIPELINE
# ─────────────────────────────────────────────
def process_loan_application(pipe, user_input: str, rules: list[dict]) -> dict:
    """
    Full Level-2 pipeline:
      Extract variables → Evaluate all rules → Apply priority →
      Fallback if needed → LLM explains → Structured output
    """
    # 1. Extract variables from free text
    variables = extract_variables(user_input)

    # 2. Fallback: nothing could be extracted
    if not variables:
        return build_fallback(user_input, variables)

    # 3. Select best matching rule with priority strategy
    rule, needs_info = select_rule(variables, rules)

    # 4. Fallback: no rule matched or a required variable was missing
    if rule is None:
        return build_fallback(user_input, variables)

    # 5. LLM writes the explanation (never changes the decision)
    reasoning = get_llm_explanation(pipe, user_input, variables, rule)

    return {
        "user_input":    user_input,
        "amount":        variables.get("amount"),
        "credit_score":  variables.get("credit_score"),
        "section":       rule["section"],
        "matched_rule":  rule["label"],
        "condition_met": rule["condition_str"],
        "decision":      rule["decision"],      # ← Set by Python rule engine
        "risk_level":    rule["risk_level"],    # ← Set by Python rule engine
        "reasoning":     reasoning,             # ← Generated by LLM
    }


def print_result(result: dict):
    icons = {
        "APPROVED":              "✅",
        "MANUAL_REVIEW":         "🔍",
        "REJECTED":              "❌",
        "NEED_MORE_INFORMATION": "❓",
    }
    icon  = icons.get(result["decision"], "")
    amt   = f"${result['amount']:,.0f}"      if result.get("amount")        else "N/A"
    score = str(int(result["credit_score"])) if result.get("credit_score")  else "N/A"

    print("=" * 65)
    print(f"  Input        : {result['user_input']}")
    print(f"  Amount       : {amt}")
    print(f"  Credit Score : {score}")
    print(f"  Section      : {result['section']}")
    print(f"  Rule         : {result['matched_rule']}")
    print(f"  Condition    : {result['condition_met']}")
    print(f"  Decision     : {icon} {result['decision']}")
    print(f"  Risk Level   : {result['risk_level']}")
    print(f"  Reasoning    :")
    print(f"    {result['reasoning']}")
    print("=" * 35 + "\n")

# ─────────────────────────────────────────────
# STEP 8: USER QUERY FUNCTION
# ─────────────────────────────────────────────

# Load rules from DataFrame
_rules = load_rules_from_df(df)

# Build pipeline from your already-loaded model and tokenizer
_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=150,
    do_sample=False,
)


def user_query(query: str):
    """
    Process a single loan request query end-to-end.

    Args:
        query: Natural language loan request including amount and optionally credit score.
               e.g. "I need a loan of 25000. My credit score is 680."

    Usage:
        user_query("I need a loan of 3000.")
        user_query("I want 25000 loan, my credit score is 720.")
        user_query("Apply for 50000, score is 580.")
    """
    result = process_loan_application(_pipe, query, _rules)
    print_result(result)

# ─────────────────────────────────────────────
# TEST CASES — run each line individually
# ─────────────────────────────────────────────

# ✅ APPROVED (3 cases)
# user_query("I need a loan of 3000 for furniture.")
# user_query("Can I borrow $1500? My credit score is 750.")
# user_query("I want a loan of 15000 and my credit score is 720.")

# 🔍 MANUAL_REVIEW (3 cases)
# user_query("I need 10000 for medical bills. My score is 650.")
# user_query("Requesting a loan of 25000, credit score is 710.")
# user_query("I want 18000 loan, score is 580.")

# ❌ REJECTED (3 cases)
# user_query("I need 30000 but my credit score is 620.")
# user_query("Apply for 50000 loan. My credit score is 550.")
# user_query("I want 200000 for real estate.")

# ❓ NEED_MORE_INFORMATION (fallback cases)
# user_query("I need a loan please.")
# user_query("My credit score is 700.")

user_query("I need a loan of 3000 for furniture.")

user_query("Can I borrow $1500? My credit score is 750.")

user_query("I need 30000 but my credit score is 620.")

user_query("Apply for 50000 loan. My credit score is 550.")

user_query("I want 200000 for real estate.")

user_query("I need a loan please.")

user_query("My credit score is 700.")

user_query("I want to buy new car and its price is $16500 and my score is 921")

user_query("I need a big loan but my credit is bad.")



"""### AND Logic
Conditions are split on the AND keyword, parsed into individual fragments, and evaluated one by one. All fragments must return True for a rule to match. If any fragment fails, the rule is immediately dismissed. No eval() is used at any point.
### Rule Priority Strategy
All rules are evaluated and every match is collected. The highest risk-level match wins — High beats Medium, Medium beats Low. Ties are broken by row order in the DataFrame. This ensures the most conservative decision always prevails.
### Fallback Behavior
NEED_MORE_INFORMATION is returned when the input contains no extractable variables, or when no rule produces a definitive match. The LLM is skipped entirely in this case and the applicant is prompted to resubmit with complete details.
### System Limitations
Variable extraction depends on keyword proximity and can misclassify ambiguous phrasing. OR conditions and nested logic are not supported. The priority strategy always escalates to the strictest outcome, which may be overly conservative for some use cases. LLM explanations are non-deterministic in wording, which may be a concern in regulated environments.
"""

