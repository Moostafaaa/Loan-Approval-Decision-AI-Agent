# Loan Approval Decision Agent
### Hybrid Rule-Based + LLM Explanation System

> ⚠️ The LLM **never** decides the loan. Python rule logic decides first. The LLM only explains.

---

## Overview

This project implements a two-level hybrid loan approval system that combines a deterministic rule engine with a large language model. The rule engine evaluates structured loan rules from a DataFrame and produces a final decision. The LLM (Phi-3.5-mini-instruct, 4-bit quantized) receives that decision and generates a professional plain-language explanation for the applicant.

---

## Project Structure

```
loan-approval-agent/
│
├── notebooks/
│   └── Loan_Approval_Decision_Agent.ipynb   # Main Colab notebook (Level 1 + Level 2)
│
├── data/
│   └── loan_rules.csv                       # Rule definitions (section, rule_description,
│                                            #   condition, decision, risk_level)
│
├── src/
│   ├── level1_engine.py                     # Level 1: single-variable rule engine
│   └── level2_engine.py                     # Level 2: multi-variable + AND logic engine
│
├── docs/
│   └── architecture.md                      # System design explanation
│       
├── requirements.txt                         # Python dependencies
└── README.md
```

---

## Levels

### Level 1 — Basic Hybrid Decision Engine
- Extracts loan `amount` from free-text input using regex
- Matches amount against single-condition rules (`<=`, `<`, `>`, `>=`, `=`, `!=`)
- First-match-wins rule selection
- LLM explains the decision

### Level 2 — Extended Hybrid Decision Engine
- Extracts multiple variables: `amount` + `credit_score`
- Supports `AND` conditions and chained range conditions (`5000 < amount <= 20000`)
- Priority strategy: **highest risk level wins** (`High > Medium > Low`), ties resolved by row order
- Fallback: returns `NEED_MORE_INFORMATION` when variables are missing or no rule matches

---

## Rule Schema

Rules are loaded from a pandas DataFrame (or CSV) with the following columns:

| Column | Description | Example |
|---|---|---|
| `section` | Loan category | `Loan` |
| `rule_description` | Human-readable rule name | `Small loan auto approved` |
| `condition` | Condition string | `amount <= 5000` or `amount > 20000 AND credit_score < 700` |
| `decision` | Output decision | `APPROVED` / `MANUAL_REVIEW` / `REJECTED` |
| `risk_level` | Risk classification | `Low` / `Medium` / `High` |

---

## Rule Priority Strategy

When multiple rules match the same input, the system applies the following strategy:

1. **Highest risk level wins** — `High` > `Medium` > `Low`
2. **Tie-break** — first matching rule by DataFrame row order wins

This ensures the most conservative, risk-aware decision always prevails.

---

## AND Logic Implementation

Compound conditions are split on the `AND` keyword before evaluation. Each fragment is parsed independently into a structured dict — either a simple condition (`amount <= 5000`) or a chained range (`5000 < amount <= 20000`). All fragments must return `True` for the rule to match. The moment any fragment fails, the rule is dismissed. `eval()` is never used.

---

## Fallback Behavior

`NEED_MORE_INFORMATION` is returned when:
- The input contains no extractable variables (no amount or credit score found)
- Variables were extracted but no rule produced a definitive match

The LLM is bypassed entirely in fallback cases.

---

## Decisions

| Decision | Meaning |
|---|---|
| ✅ `APPROVED` | Loan approved by rule engine |
| 🔍 `MANUAL_REVIEW` | Requires human review |
| ❌ `REJECTED` | Loan rejected by rule engine |
| ❓ `NEED_MORE_INFORMATION` | Missing variables or no rule matched |

---

## System Limitations

- Variable extraction relies on keyword proximity heuristics and may misclassify ambiguous phrasing
- Only `AND` logic is supported — `OR` conditions and nested logic are not
- The priority strategy always escalates to the strictest outcome, which may be overly conservative
- LLM explanations are non-deterministic in wording, which may be a concern in regulated environments

---

## Model

**Phi-3.5-mini-instruct** loaded locally from Google Drive with 4-bit quantization (NF4, float16).

```python
model_path = "/content/drive/MyDrive/Phi_3_5_mini_instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
```

---

## Usage

```python
# Single query — end-to-end pipeline
user_query("I need a loan of 25000. My credit score is 680.")
```

Example output:
```
=================================================================
  Input        : I need a loan of 25000. My credit score is 680.
  Amount       : $25,000
  Credit Score : 680
  Section      : Loan
  Rule         : High loan with low credit rejected
  Condition    : amount > 20000 AND credit_score < 700
  Decision     : ❌ REJECTED
  Risk Level   : High
  Reasoning    :
    Your application has been declined as the requested amount
    exceeds $20,000 and your credit score is below the 700
    threshold required for loans of this size.
=================================================================
```

---

## Requirements

```
torch
transformers
accelerate
bitsandbytes
pandas
```

Install with:
```bash
pip install torch transformers accelerate bitsandbytes pandas
```

---

## Architecture

```
User Input
    ↓
Extract Variables (amount, credit_score)
    ↓
Parse AND Conditions
    ↓
Evaluate All Rules (no eval())
    ↓
Apply Priority Strategy (High > Medium > Low)
    ↓
Select Final Rule  ──── No match? → NEED_MORE_INFORMATION
    ↓
LLM Explanation (Phi-3.5-mini-instruct)
    ↓
Structured Output
```
