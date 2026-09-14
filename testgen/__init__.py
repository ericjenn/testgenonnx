"""testgen — equivalence-class-based test generator for (S)ONNX operators.

Implements the strategy described in guidelines/strategy.md
(see guidelines/test-strategy-understanding.md for the recorded
understanding). Only the equivalence-class-based strategy is implemented;
mutation testing of the retained test sets is included as the validation
of its pruning decisions.
"""

__version__ = "0.1.0"