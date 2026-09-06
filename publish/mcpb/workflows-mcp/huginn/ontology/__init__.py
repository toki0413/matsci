"""Action ontology: formal action types with preconditions, effects, constraints.

Inspired by the insight that the core question is not "what answer is most
likely correct" but "what action is executable, verifiable, traceable, and
controllable under constraints."

Each action type carries:
  - preconditions: what must be true before execution
  - effects: what changes after execution (positive/negative)
  - constraints: bounds that must hold during and after
  - verifiability: how to check the action succeeded
  - traceability: audit trail fields

The predictability of each action (from the PNAS spin-glass framework)
decomposes into local contributions — the preconditions and constraints
each contribute a factor to the overall action predictability score.
"""
