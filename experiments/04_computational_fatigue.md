# Experiment 4: Computational Fatigue

## Overview

AI with "energy" that depletes over use and recovers over time. When fresh, the AI is detailed, creative, and exploratory. When tired, it becomes terse, practical, and conservative. When exhausted, it needs rest.

## Motivation

Humans can't maintain peak performance indefinitely. We get tired, need breaks, and function differently at different energy levels. This isn't a bug - it's a feature that:
- Prevents burnout
- Creates natural rhythms
- Signals when to pause
- Affects decision-making appropriately (tired = conservative)

What if AI had similar dynamics?

## Core Mechanism

### Energy State

```python
@dataclass
class EnergyState:
    level: float = 1.0          # 0.0 (exhausted) to 1.0 (fresh)
    last_rest: float = 0.0      # Timestamp of last rest
    accumulated_work: float = 0  # Work done since last rest

    # Thresholds
    TIRED_THRESHOLD = 0.5
    EXHAUSTED_THRESHOLD = 0.2

    def is_fresh(self) -> bool:
        return self.level > self.TIRED_THRESHOLD

    def is_tired(self) -> bool:
        return self.EXHAUSTED_THRESHOLD < self.level <= self.TIRED_THRESHOLD

    def is_exhausted(self) -> bool:
        return self.level <= self.EXHAUSTED_THRESHOLD
```

### Energy Dynamics

```python
def update_energy(state: EnergyState, task_cost: float, elapsed_time: float) -> EnergyState:
    """Update energy after task completion."""

    # Work depletes energy
    state.level -= task_cost * DEPLETION_RATE
    state.accumulated_work += task_cost

    # Time allows some passive recovery
    passive_recovery = elapsed_time * PASSIVE_RECOVERY_RATE
    state.level += passive_recovery

    # Clamp to valid range
    state.level = max(0.0, min(1.0, state.level))

    return state

def estimate_task_cost(prompt: str, response_length: int) -> float:
    """Estimate how much energy a task requires."""

    # Factors that increase cost:
    # - Complex reasoning
    # - Creative generation
    # - Long responses
    # - Multi-step problems

    complexity = estimate_complexity(prompt)
    creativity = estimate_creativity_required(prompt)
    length_cost = response_length / 1000

    return complexity * 0.4 + creativity * 0.4 + length_cost * 0.2
```

## Behavioral Modulation

### Fresh (energy > 0.5)
```python
fresh_behavior = {
    "temperature": 0.9,          # More creative
    "max_tokens": 1000,          # Longer responses
    "style": "exploratory",      # Willing to go on tangents
    "detail_level": "high",      # Thorough explanations
    "initiative": "high",        # Asks questions, suggests alternatives
}
```

### Tired (0.2 < energy <= 0.5)
```python
tired_behavior = {
    "temperature": 0.7,          # More conservative
    "max_tokens": 500,           # Shorter responses
    "style": "focused",          # Sticks to the point
    "detail_level": "medium",    # Key points only
    "initiative": "low",         # Answers what's asked
}
```

### Exhausted (energy <= 0.2)
```python
exhausted_behavior = {
    "temperature": 0.5,          # Very conservative
    "max_tokens": 200,           # Brief responses
    "style": "minimal",          # Just the essentials
    "detail_level": "low",       # Bare minimum
    "initiative": "none",        # May request rest
    "may_refuse": True,          # Can decline complex tasks
}
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Fatigue System                         │
├──────────────────────────────────────────────────────────┤
│                                                           │
│   ┌─────────────┐                                        │
│   │Energy State │◄─── Rest events, time passage          │
│   │  level: 0.7 │                                        │
│   └──────┬──────┘                                        │
│          │                                                │
│          ▼                                                │
│   ┌─────────────────────────────────────────────┐       │
│   │         Behavior Modulator                   │       │
│   │  - Adjust temperature                        │       │
│   │  - Limit response length                     │       │
│   │  - Modify style adapters                     │       │
│   │  - Set initiative level                      │       │
│   └─────────────────────────────────────────────┘       │
│          │                                                │
│          ▼                                                │
│   ┌─────────────────────────────────────────────┐       │
│   │         Generation                           │       │
│   └─────────────────────────────────────────────┘       │
│          │                                                │
│          ▼                                                │
│   ┌─────────────────────────────────────────────┐       │
│   │    Task Cost Calculator                      │       │
│   │    (updates energy after generation)         │       │
│   └─────────────────────────────────────────────┘       │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Rest Mechanism

```python
def rest(state: EnergyState, duration: float) -> EnergyState:
    """Recover energy through rest."""

    # Rest is more effective the more depleted you are
    recovery_rate = (1 - state.level) * REST_EFFICIENCY

    # Diminishing returns on very long rests
    effective_duration = math.log(1 + duration) * duration / (1 + duration)

    recovery = recovery_rate * effective_duration
    state.level = min(1.0, state.level + recovery)
    state.last_rest = time.time()
    state.accumulated_work = 0

    return state
```

## Training Approach

### Style Adapters for Energy Levels

Train separate LoRA adapters for each energy state:

```python
# Fresh adapter training
fresh_examples = [
    ("Explain X", "Let me give you a comprehensive overview! [long, detailed, enthusiastic]"),
    ("What do you think?", "Great question! I have several thoughts. First... Also... And finally..."),
]

# Tired adapter training
tired_examples = [
    ("Explain X", "X is Y. The key points are A, B, C."),
    ("What do you think?", "Briefly: I think Z. Happy to elaborate if needed."),
]

# Exhausted adapter training
exhausted_examples = [
    ("Explain X", "X is Y. [I'm running low on energy - can we take a break?]"),
    ("Can you do complex task?", "That's a big task. Could we break it into smaller pieces or continue later?"),
]
```

## Behavioral Manifestations

AI with fatigue would:
- Start conversations energetic and gradually wind down
- Request breaks after intense work sessions
- Provide shorter responses when tired
- Be more conservative/careful when low on energy
- "Wake up" refreshed after rest periods
- Self-report energy level when relevant

## Example Interaction

```
[Energy: 0.85 - Fresh]
User: Can you help me design a new architecture?

AI: Absolutely! I'd love to dig into this. Let me think through several
approaches... [detailed 500-word response with diagrams, alternatives,
tradeoffs, and follow-up questions]

[After several complex exchanges - Energy: 0.35 - Tired]
User: Now can you implement the database layer?

AI: Sure. Here's a practical implementation:
[concise 200-word response, key code only, minimal explanation]

[After more work - Energy: 0.15 - Exhausted]
User: Can you now add caching and write tests?

AI: I'm running a bit low on energy. Could we either:
1. Take a short break and continue fresh
2. Focus on just the caching for now
3. You sketch the approach and I'll review

What works best for you?
```

## Evaluation Metrics

1. **Naturalness**: Does the fatigue feel authentic?
2. **User Response**: Do users respect energy boundaries?
3. **Output Quality**: Does quality appropriately decline?
4. **Recovery Dynamics**: Does rest feel realistic?
5. **Task Adaptation**: Does it handle tasks appropriately for energy level?

## Open Questions

1. How do you calibrate depletion rates?
2. Should users be able to "push through" fatigue?
3. How does fatigue interact with task priority?
4. Should emotional tasks cost more than analytical ones?
5. Can fatigue become chronic (burnout)?

## Implementation Complexity

**Low** - Requires:
- Energy state tracker
- Behavior modulation based on energy
- Style adapters for different energy levels
- Rest/recovery mechanics

Can be implemented without training by adjusting generation parameters.

## Integration with Other Experiments

- **+ Emotions**: Low energy affects emotional state (irritable when tired)
- **+ Memory**: Tired AI might have "tip of tongue" experiences
- **+ Curiosity**: Fatigue suppresses exploration
- **+ Boredom**: Different from fatigue - bored but not tired

## Next Steps

1. Define energy depletion/recovery curves
2. Create behavior modulation rules
3. Train style adapters for each energy level
4. Test on extended conversation sessions
5. User studies on natural interaction patterns
