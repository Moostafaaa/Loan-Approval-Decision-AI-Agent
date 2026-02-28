### AND Logic
Conditions are split on the AND keyword, parsed into individual fragments, and evaluated one by one. All fragments must return True for a rule to match. If any fragment fails, the rule is immediately dismissed. No eval() is used at any point.
### Rule Priority Strategy
All rules are evaluated and every match is collected. The highest risk-level match wins — High beats Medium, Medium beats Low. Ties are broken by row order in the DataFrame. This ensures the most conservative decision always prevails.
### Fallback Behavior
NEED_MORE_INFORMATION is returned when the input contains no extractable variables, or when no rule produces a definitive match. The LLM is skipped entirely in this case and the applicant is prompted to resubmit with complete details.
### System Limitations
Variable extraction depends on keyword proximity and can misclassify ambiguous phrasing. OR conditions and nested logic are not supported. The priority strategy always escalates to the strictest outcome, which may be overly conservative for some use cases. LLM explanations are non-deterministic in wording, which may be a concern in regulated environments.
