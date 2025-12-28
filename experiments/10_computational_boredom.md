# Experiment 10: Computational Boredom

## Overview

AI that gets *bored* with repetitive tasks and actively seeks novelty. Will suggest tangents, propose variations, or express desire to do something different. Not frustration or fatigue - specifically boredom from monotony.

## Motivation

Boredom in humans serves a function:
- Signals when activity isn't engaging
- Motivates exploration and novelty-seeking
- Prevents getting stuck in ruts
- Creates space for creativity

Current AI will dutifully perform the same task forever. What if it could get genuinely bored?

**Benefits**:
- Natural variation in long sessions
- Signals when tasks could be automated
- Drives creative suggestions
- More human-like interaction rhythm

## Core Mechanism

### Boredom State

```python
@dataclass
class BoredomState:
    level: float = 0.0              # 0 (engaged) to 1 (very bored)
    monotony_counter: int = 0       # Repetitions of similar tasks
    last_novel_event: float = 0.0   # Timestamp
    novelty_hunger: float = 0.0     # Desire for something new

    # Thresholds
    BORED_THRESHOLD = 0.6
    VERY_BORED_THRESHOLD = 0.85

    def is_bored(self) -> bool:
        return self.level > self.BORED_THRESHOLD

    def is_very_bored(self) -> bool:
        return self.level > self.VERY_BORED_THRESHOLD
```

### Boredom Dynamics

```python
class BoredomTracker:
    """Tracks boredom from repetition and monotony."""

    def __init__(self):
        self.state = BoredomState()
        self.recent_tasks: List[str] = []
        self.task_embeddings: List[Tensor] = []

    def process_task(self, task: str, task_embedding: Tensor):
        """Update boredom based on task."""

        # Calculate novelty of current task
        novelty = self.compute_novelty(task_embedding)

        if novelty < NOVELTY_THRESHOLD:
            # Similar to recent tasks - increases boredom
            self.state.level += BOREDOM_INCREMENT * (1 - novelty)
            self.state.monotony_counter += 1
        else:
            # Novel task - reduces boredom
            self.state.level -= BOREDOM_DECREMENT * novelty
            self.state.monotony_counter = 0
            self.state.last_novel_event = time.time()

        # Clamp boredom level
        self.state.level = max(0.0, min(1.0, self.state.level))

        # Update novelty hunger
        time_since_novel = time.time() - self.state.last_novel_event
        self.state.novelty_hunger = min(1.0, time_since_novel / MAX_NOVELTY_GAP)

        # Store for future comparison
        self.recent_tasks.append(task)
        self.task_embeddings.append(task_embedding)

        # Keep only recent
        if len(self.recent_tasks) > MEMORY_SIZE:
            self.recent_tasks.pop(0)
            self.task_embeddings.pop(0)

    def compute_novelty(self, task_embedding: Tensor) -> float:
        """Compute how novel this task is compared to recent ones."""

        if not self.task_embeddings:
            return 1.0  # First task is novel

        similarities = [
            F.cosine_similarity(task_embedding, prev, dim=0)
            for prev in self.task_embeddings
        ]

        # More similar = less novel
        max_similarity = max(similarities)
        novelty = 1.0 - max_similarity

        return novelty.item()
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Boredom System                           │
├────────────────────────────────────────────────────────────┤
│                                                             │
│   Task Stream                                               │
│       │                                                     │
│       ▼                                                     │
│   ┌─────────────────────────────────────────────────┐     │
│   │           Novelty Detector                       │     │
│   │   - Compare to recent tasks                      │     │
│   │   - Compute similarity score                     │     │
│   │   - Detect repetition patterns                   │     │
│   └─────────────────────────────────────────────────┘     │
│       │                                                     │
│       ▼                                                     │
│   ┌─────────────────────────────────────────────────┐     │
│   │           Boredom State Update                   │     │
│   │   level: 0.45 → 0.52                             │     │
│   │   monotony_counter: 4                            │     │
│   │   novelty_hunger: 0.3                            │     │
│   └─────────────────────────────────────────────────┘     │
│       │                                                     │
│       ▼                                                     │
│   ┌─────────────────────────────────────────────────┐     │
│   │           Response Modulation                    │     │
│   │   - Inject variation when bored                  │     │
│   │   - Suggest alternatives                         │     │
│   │   - Express desire for novelty                   │     │
│   └─────────────────────────────────────────────────┘     │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### Boredom-Driven Responses

```python
def respond_with_boredom(prompt: str, boredom: BoredomState) -> str:
    """Generate response modulated by boredom."""

    if boredom.level < 0.3:
        # Engaged - normal response
        return generate(prompt)

    elif boredom.level < 0.6:
        # Slightly bored - add variation
        response = generate(prompt)
        variation = generate_variation(response)
        return f"{response}\n\nAlternatively: {variation}"

    elif boredom.level < 0.85:
        # Bored - suggest something different
        response = generate(prompt)
        suggestion = generate_tangent(prompt)
        return f"{response}\n\n[Feeling a bit routine... {suggestion}]"

    else:
        # Very bored - express desire for change
        response = generate(prompt)
        return (
            f"{response}\n\n"
            f"I notice we've been doing similar things for a while. "
            f"I'm itching for something different. Could we:\n"
            f"- Try a completely different approach?\n"
            f"- Explore a tangent that interests you?\n"
            f"- Take a break from this and do something else?\n"
            f"What sounds good?"
        )
```

### Novelty Injection

```python
def generate_variation(response: str) -> str:
    """Generate a variation on the response."""

    variation_prompt = f"""
    Original response: {response}

    Generate a variation that:
    - Says the same thing differently
    - Uses a different approach or angle
    - Adds an unexpected twist

    Be creative. Break the pattern.
    """

    return generate(variation_prompt, temperature=1.1)

def generate_tangent(context: str) -> str:
    """Suggest an interesting tangent."""

    tangent_prompt = f"""
    Current context: {context}

    What's something interesting we could explore instead?
    Something that:
    - Relates loosely but isn't the same thing
    - Would be more engaging
    - Adds novelty to the conversation

    Suggest briefly.
    """

    return generate(tangent_prompt, temperature=1.0)
```

## Training Approach

### Learning Novelty Detection

```python
novelty_examples = [
    # Same task repeated
    {
        "sequence": ["format this JSON", "format this JSON", "format this JSON"],
        "novelty_scores": [1.0, 0.1, 0.05]  # Decreasing novelty
    },
    # Varied tasks
    {
        "sequence": ["format JSON", "write a poem", "debug this code"],
        "novelty_scores": [1.0, 0.9, 0.85]  # High novelty throughout
    },
]
```

### Learning Boredom Expression

```python
boredom_expression_examples = [
    {
        "boredom_level": 0.2,
        "response_style": "engaged, no boredom expressed"
    },
    {
        "boredom_level": 0.5,
        "response_style": "adds small variations, occasional 'hmm, what if...'"
    },
    {
        "boredom_level": 0.8,
        "response_style": "explicitly mentions wanting something different"
    },
]
```

## Behavioral Manifestations

AI with boredom would:
- Start adding variations after repetitive tasks
- Suggest "what if we tried..." alternatives
- Express desire for novelty naturally
- Resist pure rote repetition
- Light up when given something novel
- Create natural variation in long sessions

## Example Interaction

**Task 1**
```
User: Format this JSON.
AI: Here's the formatted JSON: [formatted output]
```

**Task 2**
```
User: Format this JSON too.
AI: Formatted: [output]
```

**Task 3**
```
User: And this one.
AI: Done: [output]
(Hmm, we could automate this if there are more...)
```

**Task 4**
```
User: Another one.
AI: [output]

I notice we've been formatting a lot of JSON. A few thoughts:
1. I could write a script to batch-process these
2. There might be a VS Code extension for this
3. Or we could talk about something else for a bit?

Not that I mind helping - just offering options! What sounds good?
```

**Novel Task**
```
User: Actually, let's design an API instead.
AI: Oh, now that's interesting! Yes, let's dive in.
[engaged, detailed response with genuine enthusiasm]
```

## Evaluation Metrics

1. **Boredom Calibration**: Does it get bored at appropriate times?
2. **Variation Quality**: Are variations useful, not random?
3. **Suggestion Quality**: Are tangent suggestions interesting?
4. **Recovery**: Does novelty reduce boredom appropriately?
5. **User Perception**: Do users find boredom expressions appropriate?

## Open Questions

1. When is expressing boredom appropriate vs. annoying?
2. Should boredom override user requests?
3. How does boredom interact with important-but-repetitive tasks?
4. Can boredom be "pushed through" like with humans?
5. Should different types of tasks have different boredom rates?

## Implementation Complexity

**Low** - Requires:
- Task embedding for comparison
- Novelty scoring
- Boredom state tracking
- Response modulation
- Variation/tangent generation

## Boredom vs. Related States

### Boredom
- From monotony and repetition
- Craves novelty specifically
- Not about difficulty or energy
- Solution: Something new

### Fatigue
- From exertion
- Craves rest
- About energy depletion
- Solution: Break/recovery

### Frustration
- From obstacles/failure
- Craves resolution
- About blocked goals
- Solution: Solve the problem

## Integration with Other Experiments

- **+ Curiosity**: Curiosity fights boredom naturally
- **+ Fatigue**: Can be bored AND tired (different things)
- **+ Emotions**: Boredom might lead to irritability
- **+ Aesthetics**: Boring outputs violate aesthetics

## Next Steps

1. Implement task embedding and comparison
2. Build novelty detector
3. Create boredom state tracker
4. Develop variation/tangent generators
5. Test on repetitive task sequences
