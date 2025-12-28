# Experiment 2: Memory with Decay & Reinforcement

## Overview

Human-like memory dynamics for AI - not just retrieval (RAG), but memories that fade, strengthen through use, form associations, and have emotional valence affecting recall.

## Motivation

Current AI memory is binary - either in context or not. Human memory is dynamic:
- Unused memories fade
- Accessed memories strengthen
- Related memories activate together
- Emotional memories are more persistent
- Memories consolidate and abstract over time

This creates AI that genuinely "remembers" rather than "retrieves."

## Core Mechanism

### Memory Entry

```python
@dataclass
class Memory:
    id: str
    content: str
    embedding: Tensor

    # Dynamic properties
    strength: float = 1.0           # Decays over time, reinforced on access
    last_accessed: float = 0.0      # Timestamp
    access_count: int = 0

    # Emotional valence
    emotional_weight: float = 0.0   # -1 (painful) to 1 (joyful)
    importance: float = 0.5         # Subjective importance

    # Associations
    associations: Dict[str, float] = field(default_factory=dict)  # id -> strength
```

### Decay Function

```python
def compute_memory_strength(memory: Memory, current_time: float) -> float:
    """Ebbinghaus-inspired forgetting curve with modifications."""

    time_delta = current_time - memory.last_accessed

    # Base decay (exponential forgetting)
    base_decay = math.exp(-time_delta / DECAY_CONSTANT)

    # Reinforcement from repeated access (spaced repetition effect)
    repetition_bonus = math.log(1 + memory.access_count) * 0.1

    # Emotional memories decay slower
    emotional_persistence = 1 + abs(memory.emotional_weight) * 0.5

    # Importance modifier
    importance_modifier = 0.5 + memory.importance

    strength = memory.strength * base_decay * emotional_persistence * importance_modifier
    strength += repetition_bonus

    return min(1.0, max(0.0, strength))
```

### Association Formation

```python
def form_association(memory_a: Memory, memory_b: Memory, context_strength: float):
    """Strengthen association between co-activated memories."""

    # Hebbian learning: neurons that fire together wire together
    current = memory_a.associations.get(memory_b.id, 0.0)

    # Association strength increases with co-activation
    new_strength = current + LEARNING_RATE * context_strength * (1 - current)

    memory_a.associations[memory_b.id] = new_strength
    memory_b.associations[memory_a.id] = new_strength
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Dynamic Memory System                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Memory    │◄──►│ Association │◄──►│   Memory    │      │
│  │   Store     │    │    Graph    │    │   Store     │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                  │                  │              │
│         ▼                  ▼                  ▼              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Recall Engine                       │    │
│  │   - Query embedding                                  │    │
│  │   - Strength-weighted retrieval                      │    │
│  │   - Association spreading activation                 │    │
│  │   - Emotional priming                                │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               Memory Consolidation                   │    │
│  │   - Compress similar memories                        │    │
│  │   - Abstract patterns                                │    │
│  │   - Prune weak memories                              │    │
│  │   - Strengthen associations                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Recall Algorithm

```python
def recall(query: str, memory_store: MemoryStore,
           emotional_context: float = 0.0) -> List[Memory]:
    """Recall memories with dynamic weighting."""

    query_embedding = embed(query)
    current_time = time.time()

    candidates = []
    for memory in memory_store.all():
        # Compute current strength
        strength = compute_memory_strength(memory, current_time)

        # Skip very weak memories
        if strength < RECALL_THRESHOLD:
            continue

        # Semantic similarity
        similarity = cosine_similarity(query_embedding, memory.embedding)

        # Emotional congruence (mood-congruent recall)
        emotional_match = 1 + emotional_context * memory.emotional_weight * 0.3

        # Final recall score
        score = similarity * strength * emotional_match

        candidates.append((memory, score))

    # Sort by score
    candidates.sort(key=lambda x: -x[1])

    # Spreading activation: also retrieve associated memories
    top_memories = [m for m, s in candidates[:5]]
    associated = spread_activation(top_memories, memory_store)

    # Reinforce accessed memories
    for memory in top_memories:
        reinforce_memory(memory, current_time)

    return top_memories + associated
```

## Training Approach

### Learning What to Remember

Not all information is worth remembering. Train a network to predict importance:

```python
class ImportancePredictor(nn.Module):
    """Predicts how important a piece of information is to remember."""

    def forward(self, content_embedding, context_embedding):
        # Factors: novelty, relevance, emotional weight, user emphasis
        features = torch.cat([content_embedding, context_embedding], dim=-1)
        importance = self.predictor(features)
        return importance
```

### Learning Associations

Track which memories are frequently co-accessed and strengthen connections.

## Behavioral Manifestations

AI with dynamic memory would:
- Forget minor details from long-ago conversations
- Remember emotionally significant moments vividly
- Make unexpected connections ("This reminds me of that thing from last month...")
- Have "tip of the tongue" moments (weak but present memories)
- Show genuine surprise when reminded of forgotten things
- Build up rich context about frequently discussed topics

## Example Interaction

```
User: Remember that caching issue we discussed?

AI: The caching issue... [searching] I remember we talked about caching
a few weeks ago. Something about LRU? The details are fuzzy, but I
distinctly remember you were frustrated because it kept failing under
load. That emotional context is still vivid. Was it the connection pool
exhaustion problem?

User: Yes! That's the one.

AI: Ah, it's coming back now. We traced it to the connection pool, and
you mentioned it was similar to an issue at your previous job. I'm
curious if you ever solved it there?
```

## Evaluation Metrics

1. **Appropriate Forgetting**: Does it forget unimportant details?
2. **Emotional Retention**: Are emotional memories more persistent?
3. **Association Quality**: Are memory connections meaningful?
4. **Recall Accuracy**: When it remembers, is it correct?
5. **Natural Degradation**: Does memory feel human-like?

## Open Questions

1. How do you handle false memories (confident but wrong)?
2. Should the AI acknowledge uncertainty about memories?
3. How do you balance forgetting with user expectations?
4. Can memories become "implicit" (affect behavior without explicit recall)?
5. How do you handle contradictory memories?

## Implementation Complexity

**Medium** - Requires:
- Vector database with custom scoring
- Decay/reinforcement update system
- Association graph
- Background consolidation process
- Importance prediction network

## Integration with Other Experiments

- **+ Emotions**: Emotional state affects recall (mood-congruent memory)
- **+ Dreams**: Consolidation happens during "dream" phase
- **+ Curiosity**: Strong memories of unsatisfied curiosity
- **+ Fatigue**: Memory formation impaired when tired

## Next Steps

1. Design memory schema and storage
2. Implement decay and reinforcement functions
3. Build association graph with spreading activation
4. Train importance predictor
5. Evaluate on multi-session conversations
