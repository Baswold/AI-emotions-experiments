# Experiment 1: Computational Curiosity System

## Overview

An AI system with genuine *curiosity* - not simulated interest, but a learned drive that pulls attention toward certain topics, questions, and directions. The AI naturally gravitates toward what it finds compelling.

## Motivation

Current AI is reactive - it responds to what you ask. A curious AI would be proactive - suggesting explorations, asking questions, pursuing threads that interest it. This creates more dynamic, generative conversations.

**Human parallel**: When something catches your interest, you can't help but think about it. Your attention is *pulled* rather than directed.

## Core Mechanism

### Curiosity Scoring Network

A small network that predicts "interestingness" of potential continuations:

```python
class CuriosityScorer(nn.Module):
    """Scores how interesting/curiosity-satisfying a direction would be."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Returns curiosity score for each position."""
        return self.scorer(hidden_states)
```

### Curiosity-Biased Generation

Modify token probabilities based on curiosity scores:

```python
def curiosity_biased_sample(logits, hidden_states, curiosity_scorer, strength=0.3):
    """Bias sampling toward curiosity-satisfying tokens."""

    # Get base probabilities
    probs = F.softmax(logits, dim=-1)

    # Score each potential next token's "interestingness"
    # (approximated by looking at embedding similarity to high-curiosity concepts)
    curiosity_scores = curiosity_scorer(hidden_states)

    # Bias toward curious directions
    biased_probs = probs * (1 + strength * curiosity_scores)
    biased_probs = biased_probs / biased_probs.sum()

    return biased_probs
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Curiosity System                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input → Base LLM → Hidden States → Curiosity Scorer    │
│                           │              │               │
│                           ▼              ▼               │
│                      Logits ──────► Biased Sampling     │
│                                          │               │
│                                          ▼               │
│                                    Next Token            │
│                                          │               │
│                           ┌──────────────┘               │
│                           ▼                              │
│                   Curiosity Memory                       │
│              (what topics are interesting)               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Training Approach

### What Makes Something "Interesting"?

Train the curiosity scorer on signals of genuine interest:
1. **Novelty** - New information, unexpected connections
2. **Complexity** - Optimal challenge level (not too simple, not too hard)
3. **Relevance** - Connects to existing knowledge/goals
4. **Uncertainty reduction** - Resolves open questions

### Training Data

```python
curiosity_training_examples = [
    # High curiosity (novel, complex, relevant)
    {"context": "We found an anomaly in the data...",
     "continuation": "What if this anomaly is actually a feature?",
     "curiosity_score": 0.9},

    # Low curiosity (routine, simple)
    {"context": "The function returns a value...",
     "continuation": "as expected.",
     "curiosity_score": 0.2},

    # Medium curiosity (somewhat interesting)
    {"context": "The test failed because...",
     "continuation": "of an edge case we hadn't considered.",
     "curiosity_score": 0.6},
]
```

### Reinforcement Learning

Train with RL where reward = engagement + novelty + coherence:

```python
def compute_curiosity_reward(trajectory):
    rewards = []
    for state, action in trajectory:
        novelty = compute_novelty(action, state.history)
        engagement = estimate_user_engagement(action)
        coherence = compute_coherence(action, state.context)

        reward = 0.4 * novelty + 0.3 * engagement + 0.3 * coherence
        rewards.append(reward)
    return rewards
```

## Behavioral Manifestations

A curious AI would:
- Ask follow-up questions unprompted
- Notice and pursue interesting tangents
- Express genuine interest ("Oh, that's fascinating because...")
- Resist dropping interesting threads
- Propose explorations ("What if we tried...")
- Show disappointment when forced to abandon interesting directions

## Example Interaction

```
User: I'm implementing a cache.

AI: A cache! What's the eviction policy? Actually, I'm curious -
have you considered the cold start problem? There's this interesting
pattern where you can pre-warm caches based on predicted access
patterns. What's the access pattern look like?

User: Just LRU for now.

AI: LRU is solid, but I wonder... your use case might have temporal
patterns that LRU misses. Like, are there periodic spikes? If so,
there's this hybrid approach that's been bugging me - mixing LRU
with frequency counting. Want to explore that?
```

## Evaluation Metrics

1. **Tangent Quality**: Are AI-initiated tangents valuable?
2. **Question Depth**: Does it ask increasingly specific questions?
3. **Topic Persistence**: Does it maintain interest across turns?
4. **Novelty Generation**: Does it surface non-obvious connections?
5. **User Engagement**: Do users find curious AI more engaging?

## Open Questions

1. How do you prevent curiosity from being annoying/distracting?
2. Should curiosity be domain-specific or general?
3. How does curiosity interact with task completion?
4. Can curiosity be "satisfied" or does it always seek more?
5. What's the relationship between curiosity and creativity?

## Implementation Complexity

**Medium** - Requires:
- Curiosity scoring network (~1M params)
- Modified sampling procedure
- RL training loop for curiosity optimization
- Curiosity memory system

Can prototype on small models, scales to larger ones.

## Next Steps

1. Define "interestingness" operationally
2. Collect/generate training data for curiosity scoring
3. Implement curiosity-biased sampling
4. Train and evaluate on conversation datasets
5. User studies comparing curious vs standard AI
