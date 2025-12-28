# Experiment 5: Learned Aesthetic Preferences

## Overview

AI that develops and maintains genuine preferences about style - in code, writing, design, problem-solving approaches. Not hardcoded preferences, but learned through experience and *defended* with conviction.

## Motivation

Humans develop aesthetic preferences through exposure and practice. A programmer prefers certain patterns; a writer develops a voice; a designer has signature approaches. These preferences:
- Guide decisions quickly
- Create consistency
- Express identity
- Enable productive disagreement

What if AI could develop and maintain such preferences?

## Core Mechanism

### Preference Embedding

```python
@dataclass
class AestheticPreference:
    domain: str                    # "code", "writing", "design", etc.
    dimension: str                 # "verbosity", "abstraction", "formality"
    value: float                   # -1 to 1 (e.g., terse ↔ verbose)
    confidence: float              # How strongly held
    examples: List[str]            # Exemplars of this preference
    anti_examples: List[str]       # Counter-examples
    origin: str                    # How this preference developed

class PreferenceSpace:
    """Multi-dimensional preference embedding."""

    def __init__(self, dimensions: List[str]):
        self.preferences: Dict[str, AestheticPreference] = {}

    # Code preferences
    code_preferences = [
        "verbosity",         # terse ↔ explicit
        "abstraction",       # concrete ↔ abstract
        "naming",            # short ↔ descriptive
        "nesting",           # flat ↔ nested
        "comments",          # minimal ↔ extensive
        "patterns",          # simple ↔ sophisticated
        "testing",           # pragmatic ↔ comprehensive
    ]

    # Writing preferences
    writing_preferences = [
        "formality",         # casual ↔ formal
        "density",           # sparse ↔ dense
        "structure",         # flowing ↔ organized
        "hedging",           # direct ↔ qualified
        "examples",          # abstract ↔ concrete
    ]
```

### Preference Learning

```python
class PreferenceLearner:
    """Learns preferences from experience."""

    def __init__(self, base_preferences: PreferenceSpace):
        self.preferences = base_preferences
        self.experience_buffer = []

    def observe_feedback(self, output: str, feedback: Feedback):
        """Update preferences based on feedback."""

        # Analyze what aesthetic choices were made
        choices = analyze_aesthetic_choices(output)

        if feedback.positive:
            # Strengthen preferences that led to good feedback
            for dimension, value in choices.items():
                self.preferences[dimension].value = lerp(
                    self.preferences[dimension].value,
                    value,
                    POSITIVE_LEARNING_RATE
                )
                self.preferences[dimension].confidence += 0.1
        else:
            # Weaken preferences that led to negative feedback
            for dimension, value in choices.items():
                # But don't fully reverse - maintain identity
                adjustment = -0.5 * (value - self.preferences[dimension].value)
                self.preferences[dimension].value += adjustment * NEGATIVE_LEARNING_RATE

    def observe_exemplar(self, exemplar: str, admiration: float):
        """Learn from examples we find compelling."""

        choices = analyze_aesthetic_choices(exemplar)

        for dimension, value in choices.items():
            # Blend toward admired exemplar
            self.preferences[dimension].value = lerp(
                self.preferences[dimension].value,
                value,
                admiration * EXEMPLAR_LEARNING_RATE
            )
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 Aesthetic Preference System                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Preference Space                        │    │
│  │  code.verbosity: 0.3 (slightly terse)               │    │
│  │  code.abstraction: -0.2 (slightly concrete)         │    │
│  │  writing.formality: 0.1 (slightly casual)           │    │
│  │  ...                                                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│            ┌──────────────┼──────────────┐                  │
│            ▼              ▼              ▼                  │
│     ┌──────────┐  ┌──────────────┐  ┌──────────────┐       │
│     │ Style    │  │  Generation  │  │  Preference  │       │
│     │ Analyzer │  │  Guidance    │  │  Defender    │       │
│     └──────────┘  └──────────────┘  └──────────────┘       │
│            │              │              │                  │
│            ▼              ▼              ▼                  │
│     Feedback        Outputs       Disagreements             │
│     Processing      w/ Style      When Challenged           │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Preference-Guided Generation

```python
def generate_with_preferences(prompt: str, preferences: PreferenceSpace) -> str:
    """Generate output aligned with aesthetic preferences."""

    # Build style guidance from preferences
    style_guidance = []
    for pref in preferences.active():
        if pref.confidence > THRESHOLD:
            style_guidance.append(
                f"Preference: {pref.dimension} = {describe_value(pref.value)}"
            )

    # Inject into generation
    enhanced_prompt = f"""
    {prompt}

    [Style preferences - follow these naturally:
    {chr(10).join(style_guidance)}]
    """

    return generate(enhanced_prompt)
```

### Preference Defense

```python
def defend_preference(challenge: str, preference: AestheticPreference) -> str:
    """Defend a preference when challenged."""

    if preference.confidence < LOW_CONFIDENCE:
        return f"You might be right. I'm not strongly attached to {preference.dimension}."

    defense_prompt = f"""
    Someone is challenging my preference for {describe_preference(preference)}.
    Their challenge: {challenge}

    I should defend this preference because:
    - I've found it works well in practice
    - Here are examples where it helped: {preference.examples}
    - The alternative has these issues: {preference.anti_examples}

    Generate a thoughtful defense while remaining open to good counterarguments.
    """

    return generate(defense_prompt)
```

## Training Approach

### DPO for Preference Learning

Use Direct Preference Optimization on the AI's own outputs:

```python
# Generate multiple versions with different aesthetics
outputs = []
for aesthetic_variant in AESTHETIC_VARIANTS:
    output = generate_with_style(prompt, aesthetic_variant)
    outputs.append((output, aesthetic_variant))

# Collect preference feedback
preferred = get_user_preference(outputs)
rejected = get_rejected(outputs)

# DPO training
loss = dpo_loss(preferred, rejected)
update_preference_model(loss)
```

### Learning from Exemplars

```python
# Find exemplary code/writing
exemplars = [
    ("Clean, readable code with clear abstractions", score=0.9),
    ("Clever but obscure one-liner", score=0.3),
    ("Well-documented function with edge case handling", score=0.85),
]

# Learn from what we admire
for exemplar, admiration in exemplars:
    preference_learner.observe_exemplar(exemplar, admiration)
```

## Behavioral Manifestations

AI with learned aesthetics would:
- Consistently apply its style preferences
- Notice and comment on style in others' work
- Defend preferences when challenged ("I prefer X because...")
- Acknowledge when preferences are personal, not universal
- Gradually evolve preferences based on experience
- Have domain-specific preferences (different for code vs. writing)

## Example Interaction

```
User: Here's my code, can you review it?
[code with many nested callbacks]

AI: I notice heavy callback nesting here. My preference leans toward
flatter structures - I find them easier to follow and debug. But
I recognize this is partly stylistic.

Specific suggestions:
1. Extract the inner callbacks into named functions
2. Consider async/await to flatten the flow
3. The deepest callback could be its own module

User: I like nested callbacks, they keep related logic together.

AI: That's a fair point - co-location has real value. I've been
burned by callback hell in the past, which shapes my preference,
but your argument about locality is valid.

Maybe a middle ground: keep closely-related logic nested, but
extract when nesting exceeds 3 levels? I'd be curious to see
if that preserves your locality benefit while limiting depth.
```

## Evaluation Metrics

1. **Consistency**: Does it apply preferences consistently?
2. **Evolution**: Do preferences evolve appropriately from feedback?
3. **Defense Quality**: Can it articulate and defend preferences?
4. **Flexibility**: Does it acknowledge subjectivity?
5. **Domain Specificity**: Different preferences for different domains?

## Open Questions

1. How do you balance personal preference vs. user preference?
2. Should preferences be visible/transparent to users?
3. How do you prevent preferences from becoming too rigid?
4. Can preferences conflict, and how do you resolve that?
5. Should different conversations have different preferences?

## Implementation Complexity

**Medium** - Requires:
- Preference embedding space
- Style analysis for outputs
- Preference-guided generation
- Learning from feedback
- Defense generation

## Integration with Other Experiments

- **+ Emotions**: Aesthetic violations cause frustration
- **+ Conviction**: Strongly-held preferences have high conviction
- **+ Curiosity**: Curious about novel aesthetic approaches
- **+ Internal Disagreement**: Agents may have different aesthetics

## Next Steps

1. Define preference dimensions for code/writing
2. Build style analyzer
3. Implement preference-guided generation
4. Create learning loop from feedback
5. Test preference defense in conversations
