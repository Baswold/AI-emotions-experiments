# Experiment 6: Metacognitive Calibration

## Overview

AI with genuine uncertainty about *what it knows* - not just linguistic hedging ("I think...") but actual calibrated confidence that tracks real-world accuracy. The AI knows when it knows, knows when it doesn't, and adjusts behavior accordingly.

## Motivation

Current AI confidently produces both correct and incorrect answers. Humans have metacognition - awareness of our own knowledge states:
- "I know this cold"
- "I'm pretty sure, but could be wrong"
- "I vaguely remember something about this"
- "I have no idea"

True metacognition enables:
- Appropriate confidence communication
- Knowing when to look things up
- Avoiding confident errors
- Seeking help appropriately

## Core Mechanism

### Calibration Network

```python
class CalibrationNetwork(nn.Module):
    """Predicts confidence in model's own outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Takes hidden states and produces confidence score
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Domain-specific calibration
        self.domain_embeddings = nn.Embedding(NUM_DOMAINS, hidden_dim // 4)

    def forward(self, hidden_states: Tensor, domain: int) -> Tensor:
        """Predict calibrated confidence."""

        # Pool hidden states
        pooled = hidden_states.mean(dim=1)

        # Add domain context
        domain_emb = self.domain_embeddings(domain)
        features = torch.cat([pooled, domain_emb], dim=-1)

        confidence = self.confidence_head(features)
        return confidence
```

### Confidence Tracking

```python
@dataclass
class ConfidenceState:
    overall: float                          # 0-1 confidence
    domain_scores: Dict[str, float]         # Per-domain confidence
    uncertainty_type: str                   # "aleatory" vs "epistemic"
    should_verify: bool                     # Suggests external verification
    reasoning: str                          # Why this confidence level

class MetacognitiveTracker:
    """Tracks and learns from prediction accuracy."""

    def __init__(self):
        self.history: List[Tuple[float, bool]] = []  # (confidence, was_correct)
        self.domain_history: Dict[str, List] = {}

    def record_outcome(self, domain: str, confidence: float, was_correct: bool):
        """Record whether a confident prediction was correct."""
        self.history.append((confidence, was_correct))
        if domain not in self.domain_history:
            self.domain_history[domain] = []
        self.domain_history[domain].append((confidence, was_correct))

    def get_calibration_error(self) -> float:
        """Expected Calibration Error - how well does confidence match accuracy?"""
        if not self.history:
            return 0.0

        # Bin predictions by confidence
        bins = defaultdict(list)
        for conf, correct in self.history:
            bin_idx = int(conf * 10)  # 10 bins
            bins[bin_idx].append(correct)

        # Calculate ECE
        ece = 0.0
        for bin_idx, outcomes in bins.items():
            bin_conf = (bin_idx + 0.5) / 10  # Bin center
            bin_acc = sum(outcomes) / len(outcomes)
            bin_weight = len(outcomes) / len(self.history)
            ece += bin_weight * abs(bin_conf - bin_acc)

        return ece
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│              Metacognitive Calibration System               │
├────────────────────────────────────────────────────────────┤
│                                                             │
│   Input ──► Base LLM ──► Hidden States                     │
│                              │                              │
│                              ▼                              │
│   ┌────────────────────────────────────────────────┐       │
│   │           Calibration Network                   │       │
│   │   - Analyze hidden state patterns               │       │
│   │   - Check domain-specific history               │       │
│   │   - Detect uncertainty markers                  │       │
│   └────────────────────────────────────────────────┘       │
│                              │                              │
│                              ▼                              │
│   ┌────────────────────────────────────────────────┐       │
│   │           Confidence Assessment                 │       │
│   │   confidence: 0.73                              │       │
│   │   domain: "python_programming"                  │       │
│   │   uncertainty_type: "epistemic"                 │       │
│   │   should_verify: false                          │       │
│   └────────────────────────────────────────────────┘       │
│                              │                              │
│                              ▼                              │
│   ┌────────────────────────────────────────────────┐       │
│   │           Response Modulation                   │       │
│   │   - Adjust language confidence                  │       │
│   │   - Add caveats if needed                       │       │
│   │   - Suggest verification for low confidence     │       │
│   └────────────────────────────────────────────────┘       │
│                              │                              │
│                              ▼                              │
│                      Final Response                         │
│             (with calibrated confidence)                    │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### Confidence-Aware Generation

```python
def generate_with_metacognition(prompt: str, calibrator: CalibrationNetwork) -> str:
    """Generate response with calibrated confidence."""

    # Generate candidate response
    hidden_states = get_hidden_states(prompt)
    response = generate(prompt)

    # Assess confidence
    domain = classify_domain(prompt)
    confidence = calibrator(hidden_states, domain)

    # Modulate response based on confidence
    if confidence > 0.9:
        # High confidence - state directly
        return response

    elif confidence > 0.7:
        # Good confidence - minor hedge
        return f"{response}\n\n(I'm fairly confident about this, but verify critical details.)"

    elif confidence > 0.5:
        # Moderate confidence - explicit uncertainty
        return f"I believe {response}\n\n(My confidence here is moderate - I'd suggest double-checking.)"

    elif confidence > 0.3:
        # Low confidence - major caveats
        return f"I'm not certain, but my best guess: {response}\n\n(Please verify - I'm uncertain about this.)"

    else:
        # Very low confidence - acknowledge ignorance
        return f"I'm not sure about this. My tentative thought: {response}\n\n(Low confidence - you should definitely look this up.)"
```

## Training Approach

### Calibration Training

Train the calibration network on (prediction, outcome) pairs:

```python
def train_calibrator(calibrator, training_data):
    """Train calibration network on historical accuracy data."""

    for hidden_states, domain, was_correct in training_data:
        predicted_conf = calibrator(hidden_states, domain)

        # Binary cross-entropy: confidence should match correctness
        loss = F.binary_cross_entropy(predicted_conf, was_correct.float())

        loss.backward()
        optimizer.step()
```

### Self-Assessment Training

```python
# Generate claim with confidence
claim = model.generate(question)
confidence = model.assess_confidence(question, claim)

# Get ground truth (from knowledge base, human verification, etc.)
is_correct = verify_claim(claim)

# Train calibrator
calibrator.record_outcome(domain, confidence, is_correct)
```

### Uncertainty Detection Patterns

Train to recognize uncertainty signals in hidden states:

```python
uncertainty_patterns = {
    "conflicting_activations": "Multiple incompatible answers activated",
    "low_activation": "Weak activation on all candidate answers",
    "out_of_distribution": "Input far from training distribution",
    "recency_gap": "Information likely outdated",
}
```

## Behavioral Manifestations

AI with metacognition would:
- Express appropriate confidence naturally
- Decline to answer rather than guess when truly uncertain
- Suggest verification for uncertain claims
- Know its weak domains and say so
- Distinguish "I don't know" from "I'm uncertain"
- Track and report its own accuracy over time

## Example Interaction

```
User: What's the capital of France?

AI: Paris. [Confidence: very high]

User: What's the population of Paris in 2024?

AI: Around 2.1 million in the city proper, about 12 million in
the metropolitan area. [Confidence: moderate]

Note: These are approximate figures. Population statistics change,
and my training data has a cutoff. For current figures, check a
recent source.

User: What's the best restaurant in Paris?

AI: I can't reliably answer this. [Confidence: low]

"Best" is subjective, and restaurant quality changes frequently.
I could list some historically notable ones, but for current
recommendations, you'd want a recent review source or local
knowledge. Would general types of Parisian cuisine be helpful instead?
```

## Evaluation Metrics

1. **Calibration Error**: Does confidence match accuracy?
2. **Selective Prediction**: Does it abstain when uncertain?
3. **Domain Awareness**: Different calibration for different domains?
4. **User Trust**: Do users calibrate trust appropriately?
5. **Improvement Over Time**: Does calibration improve with feedback?

## Open Questions

1. How do you get ground truth for calibration training?
2. Should calibration be visible or implicit?
3. How do you handle novel domains with no history?
4. Can calibration transfer across different types of knowledge?
5. How do you distinguish "hard to know" from "I don't know"?

## Implementation Complexity

**High** - Requires:
- Calibration network training
- Ground truth collection for claims
- Domain classification
- Confidence-modulated generation
- Ongoing calibration tracking

## Uncertainty Types

### Epistemic Uncertainty
"I don't know, but someone could know"
- Factual gaps in training
- Out-of-distribution queries
- Appropriate response: "I don't know, but you could look it up"

### Aleatory Uncertainty
"Nobody can know for certain"
- Inherently random/unpredictable
- Future events
- Subjective matters
- Appropriate response: "This is inherently uncertain"

## Integration with Other Experiments

- **+ Internal Disagreement**: Disagreement indicates uncertainty
- **+ Memory**: Memory strength affects confidence
- **+ Conviction**: High-confidence beliefs are harder to change
- **+ Fatigue**: Tired AI might be less well-calibrated

## Next Steps

1. Build calibration network architecture
2. Collect prediction-outcome training data
3. Train calibrator on historical accuracy
4. Integrate with generation pipeline
5. Evaluate calibration quality over time
