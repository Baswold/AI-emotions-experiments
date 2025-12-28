# Experiment 3: Internal Disagreement / Multi-Agent Self

## Overview

Multiple "perspectives" within one AI that debate internally before responding. Instead of a single voice, the AI has internal agents (optimist, pessimist, pragmatist, creative, critic) that must reach some form of consensus.

## Motivation

Human thinking involves internal debate - we consider multiple angles, play devil's advocate, weigh competing values. Current AI produces a single "averaged" response. What if we made the internal disagreement explicit and productive?

**Benefits**:
- More nuanced, well-considered responses
- Natural expression of uncertainty and tradeoffs
- Resistance to manipulation (multiple agents must agree)
- Richer reasoning visible to the user

## Core Mechanism

### Internal Agents

```python
@dataclass
class InternalAgent:
    name: str
    role: str
    adapter: nn.Module  # LoRA or full adapter
    weight: float = 1.0  # Influence in consensus

AGENTS = {
    "optimist": InternalAgent(
        name="optimist",
        role="Sees opportunities, possibilities, best-case scenarios",
        adapter=load_adapter("optimist_lora")
    ),
    "skeptic": InternalAgent(
        name="skeptic",
        role="Questions assumptions, finds flaws, demands evidence",
        adapter=load_adapter("skeptic_lora")
    ),
    "pragmatist": InternalAgent(
        name="pragmatist",
        role="Focuses on practical implementation, real constraints",
        adapter=load_adapter("pragmatist_lora")
    ),
    "creative": InternalAgent(
        name="creative",
        role="Explores unusual angles, makes unexpected connections",
        adapter=load_adapter("creative_lora")
    ),
    "advocate": InternalAgent(
        name="advocate",
        role="Represents user's interests, empathizes with their position",
        adapter=load_adapter("advocate_lora")
    ),
}
```

### Debate Protocol

```python
def internal_debate(prompt: str, max_rounds: int = 3) -> Response:
    """Run internal debate between agents."""

    # Phase 1: Initial positions
    positions = {}
    for name, agent in AGENTS.items():
        with agent.adapter:
            positions[name] = generate_position(prompt, agent.role)

    # Phase 2: Debate rounds
    for round in range(max_rounds):
        # Each agent responds to others' positions
        rebuttals = {}
        for name, agent in AGENTS.items():
            others = {k: v for k, v in positions.items() if k != name}
            with agent.adapter:
                rebuttals[name] = generate_rebuttal(positions[name], others, agent.role)

        # Update positions based on rebuttals
        positions = update_positions(positions, rebuttals)

        # Check for consensus
        if consensus_reached(positions):
            break

    # Phase 3: Synthesis
    final_response = synthesize_positions(positions, prompt)

    return final_response
```

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Multi-Agent Self                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │ Optimist │  │ Skeptic  │  │Pragmatist│  │ Creative │      │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│        │             │             │             │              │
│        ▼             ▼             ▼             ▼              │
│   ┌─────────────────────────────────────────────────────┐      │
│   │                  Debate Arena                        │      │
│   │   Round 1: Initial positions                         │      │
│   │   Round 2: Rebuttals and refinements                │      │
│   │   Round 3: Convergence attempts                      │      │
│   └─────────────────────────────────────────────────────┘      │
│                           │                                     │
│                           ▼                                     │
│   ┌─────────────────────────────────────────────────────┐      │
│   │                  Synthesis                           │      │
│   │   - Identify agreements                              │      │
│   │   - Acknowledge disagreements                        │      │
│   │   - Weight by agent confidence                       │      │
│   │   - Generate unified response                        │      │
│   └─────────────────────────────────────────────────────┘      │
│                           │                                     │
│                           ▼                                     │
│                    Final Response                               │
│           (may include internal disagreement)                   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Consensus Mechanisms

```python
def synthesize_positions(positions: Dict[str, str], prompt: str) -> str:
    """Synthesize multiple perspectives into final response."""

    # Option 1: Unanimous agreement required
    if all_agree(positions):
        return generate_agreed_response(positions)

    # Option 2: Majority with dissent noted
    majority, dissent = find_majority(positions)
    response = generate_majority_response(majority)
    if dissent:
        response += f"\n\n[Internal dissent from {dissent.agent}: {dissent.concern}]"
    return response

    # Option 3: Present multiple perspectives
    return format_perspectives(positions)

    # Option 4: Weighted average based on confidence
    return weighted_synthesis(positions)
```

## Training Approach

### Training Individual Agents

Each agent is a LoRA adapter trained on role-specific data:

```python
# Skeptic training data
skeptic_examples = [
    {"prompt": "This approach will definitely work",
     "response": "Let's examine that claim. What evidence supports 'definitely'? What could go wrong?"},
    {"prompt": "Everyone uses this pattern",
     "response": "Popularity doesn't equal correctness. What are the actual tradeoffs?"},
]

# Optimist training data
optimist_examples = [
    {"prompt": "This might not work",
     "response": "Let's explore how it could work. What if we addressed the concerns?"},
    {"prompt": "This is a hard problem",
     "response": "Hard problems often have elegant solutions. What possibilities are we missing?"},
]
```

### Training Synthesis

Train a synthesis module to combine perspectives coherently:

```python
synthesis_examples = [
    {"positions": {"optimist": "...", "skeptic": "...", "pragmatist": "..."},
     "good_synthesis": "While there's potential here [optimist], we should address [skeptic's concerns] through [pragmatist's approach]",
     "bad_synthesis": "Everyone disagrees so idk"},
]
```

## Behavioral Manifestations

AI with internal disagreement would:
- Express nuanced views with acknowledged tradeoffs
- Sometimes say "Part of me thinks X, but I also see Y"
- Be harder to manipulate (multiple agents must be convinced)
- Show genuine internal conflict on difficult questions
- Occasionally produce responses with visible dissent
- Evolve positions through self-debate

## Example Interaction

```
User: Should I rewrite this codebase from scratch?

AI: [Internal debate in progress...]

My pragmatist says: Rarely worth it. Incremental improvement is
usually safer and cheaper.

My optimist counters: But sometimes a fresh start enables
architectures that are impossible to reach incrementally.

My skeptic asks: What specific problems would rewriting solve
that refactoring couldn't?

Synthesis: I lean toward incremental improvement, BUT if you have
specific architectural blockers (not just "it's messy"), a rewrite
might be justified. What's actually blocked right now?

[Dissent: Creative thinks we should explore a hybrid approach -
rewriting the core while wrapping the legacy code]
```

## Evaluation Metrics

1. **Response Nuance**: Are responses more nuanced than single-agent?
2. **Manipulation Resistance**: Is it harder to get bad outputs?
3. **Reasoning Quality**: Is the reasoning more thorough?
4. **User Preference**: Do users prefer multi-perspective responses?
5. **Consistency**: Do agents maintain consistent personalities?

## Open Questions

1. How many agents is optimal?
2. Should agents be visible to users or hidden?
3. How do you prevent agents from all converging to same view?
4. What happens when consensus is impossible?
5. Can agents learn from each other over time?

## Implementation Complexity

**High** - Requires:
- Multiple LoRA adapters (one per agent)
- Debate orchestration system
- Consensus/synthesis mechanism
- Significantly more compute (multiple forward passes)

## Variants

### Lightweight Version
Instead of separate adapters, use different system prompts:
```python
def lightweight_debate(prompt):
    perspectives = []
    for role in ["optimist", "skeptic", "pragmatist"]:
        response = generate(prompt, system=f"You are a {role}. {ROLE_DESC[role]}")
        perspectives.append(response)
    return synthesize(perspectives)
```

### Adversarial Version
One agent actively tries to break/challenge the others:
```python
adversary_role = "Find flaws in the other agents' reasoning. Be harsh but fair."
```

## Next Steps

1. Train individual agent adapters
2. Design debate protocol and consensus mechanisms
3. Build synthesis module
4. Evaluate on controversial/complex questions
5. User studies on preference and trust
