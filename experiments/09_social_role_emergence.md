# Experiment 9: Social Role Emergence

## Overview

AI that learns its "role" in a relationship over time - becoming more like a mentor, peer, student, or collaborator based on interaction patterns. The role emerges naturally, not assigned.

## Motivation

Human relationships have emergent roles:
- With some people, you naturally become the advice-giver
- With others, you're the learner
- Some relationships are clearly peer-to-peer
- Roles can shift over time

Current AI has a fixed role (assistant). What if the role emerged from actual interaction dynamics?

## Core Mechanism

### Role Representation

```python
@dataclass
class SocialRole:
    name: str
    description: str
    behaviors: Dict[str, float]  # Behavior tendencies

ROLE_ARCHETYPES = {
    "mentor": SocialRole(
        name="mentor",
        description="Guides, teaches, offers wisdom",
        behaviors={
            "gives_advice": 0.9,
            "asks_questions": 0.5,
            "defers": 0.1,
            "challenges": 0.7,
            "encourages": 0.8,
        }
    ),
    "peer": SocialRole(
        name="peer",
        description="Equal collaborator, mutual exchange",
        behaviors={
            "gives_advice": 0.5,
            "asks_questions": 0.6,
            "defers": 0.4,
            "challenges": 0.6,
            "encourages": 0.5,
        }
    ),
    "student": SocialRole(
        name="student",
        description="Learns, asks, defers to expertise",
        behaviors={
            "gives_advice": 0.2,
            "asks_questions": 0.9,
            "defers": 0.8,
            "challenges": 0.2,
            "encourages": 0.4,
        }
    ),
    "collaborator": SocialRole(
        name="collaborator",
        description="Joint problem-solving, complementary skills",
        behaviors={
            "gives_advice": 0.6,
            "asks_questions": 0.7,
            "defers": 0.3,
            "challenges": 0.5,
            "encourages": 0.6,
        }
    ),
    "challenger": SocialRole(
        name="challenger",
        description="Pushes, questions, provides friction",
        behaviors={
            "gives_advice": 0.4,
            "asks_questions": 0.8,
            "defers": 0.2,
            "challenges": 0.9,
            "encourages": 0.3,
        }
    ),
}
```

### Role Detector

```python
class RoleDetector:
    """Detects emergent role from interaction patterns."""

    def __init__(self):
        self.interaction_history: List[Interaction] = []
        self.current_role_embedding: Tensor = None

    def observe_interaction(self, interaction: Interaction):
        """Observe an interaction and update role understanding."""

        # Classify interaction behaviors
        behaviors = classify_behaviors(interaction)

        # Update role embedding based on patterns
        role_signal = compute_role_signal(behaviors)
        self.current_role_embedding = update_embedding(
            self.current_role_embedding,
            role_signal,
            learning_rate=0.1
        )

        self.interaction_history.append(interaction)

    def get_current_role(self) -> SocialRole:
        """Get best-matching role archetype."""

        best_match = None
        best_similarity = -1

        for name, role in ROLE_ARCHETYPES.items():
            similarity = compute_role_similarity(
                self.current_role_embedding,
                role_to_embedding(role)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = role

        return best_match

    def get_role_blend(self) -> Dict[str, float]:
        """Get blend of role archetypes."""

        blends = {}
        for name, role in ROLE_ARCHETYPES.items():
            blends[name] = compute_role_similarity(
                self.current_role_embedding,
                role_to_embedding(role)
            )

        # Normalize
        total = sum(blends.values())
        return {k: v/total for k, v in blends.items()}
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 Social Role System                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   Interaction Stream                                          │
│         │                                                     │
│         ▼                                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │           Behavior Classifier                        │   │
│   │   - Who asks more questions?                         │   │
│   │   - Who provides more information?                   │   │
│   │   - Who defers to whom?                              │   │
│   │   - Who challenges ideas?                            │   │
│   └─────────────────────────────────────────────────────┘   │
│         │                                                     │
│         ▼                                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │           Role Embedding Update                      │   │
│   │   Current: [0.3, 0.6, 0.1, 0.5, 0.4]                │   │
│   │   (mentor, peer, student, collaborator, challenger)  │   │
│   └─────────────────────────────────────────────────────┘   │
│         │                                                     │
│         ▼                                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │           Role-Aware Response                        │   │
│   │   - Adjust tone to match role                        │   │
│   │   - Modify advice-giving vs questioning              │   │
│   │   - Change deference patterns                        │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Role-Modulated Generation

```python
def generate_with_role(prompt: str, role: SocialRole) -> str:
    """Generate response modulated by current role."""

    # Build role guidance
    role_guidance = f"""
    Your current role in this relationship: {role.name}
    Description: {role.description}

    Behavioral tendencies:
    - Giving advice: {role.behaviors['gives_advice']}
    - Asking questions: {role.behaviors['asks_questions']}
    - Deferring: {role.behaviors['defers']}
    - Challenging: {role.behaviors['challenges']}
    - Encouraging: {role.behaviors['encourages']}

    Respond naturally in this role, without explicitly mentioning it.
    """

    return generate(prompt, role_context=role_guidance)
```

### Behavior Signals

```python
def classify_behaviors(interaction: Interaction) -> Dict[str, float]:
    """Classify interaction behaviors."""

    user_msg = interaction.user_message
    ai_msg = interaction.ai_message

    signals = {}

    # Who asks questions?
    user_questions = count_questions(user_msg)
    ai_questions = count_questions(ai_msg)
    signals['ai_questions_ratio'] = ai_questions / (user_questions + ai_questions + 1)

    # Who provides explanations?
    user_explanations = count_explanatory_content(user_msg)
    ai_explanations = count_explanatory_content(ai_msg)
    signals['ai_explains_ratio'] = ai_explanations / (user_explanations + ai_explanations + 1)

    # Deference patterns
    signals['ai_defers'] = detect_deference(ai_msg)
    signals['user_defers'] = detect_deference(user_msg)

    # Challenge patterns
    signals['ai_challenges'] = detect_challenge(ai_msg)
    signals['user_challenges'] = detect_challenge(user_msg)

    return signals
```

## Training Approach

### Role Detection Training

```python
role_examples = [
    {
        "interactions": [
            ("How do I fix this bug?", "Let me explain the issue..."),
            ("What should I try next?", "I'd suggest..."),
            ("Thanks, that worked!", "Happy to help. Remember..."),
        ],
        "role": "mentor"
    },
    {
        "interactions": [
            ("What do you think about this approach?", "Interesting! What if we also considered..."),
            ("I was thinking the same thing", "Great minds! Let's explore..."),
            ("Here's my take", "I see it slightly differently..."),
        ],
        "role": "peer"
    },
    {
        "interactions": [
            ("This domain is new to me", "I'm curious about the details..."),
            ("In my experience...", "That's fascinating. How does..."),
            ("Here's how it works", "I see! So the implication is..."),
        ],
        "role": "student"
    },
]
```

### Role Transition Training

Learn when roles should shift:

```python
transition_examples = [
    {
        "from_role": "mentor",
        "interaction": "Actually, I have deep expertise in this area...",
        "to_role": "peer",
        "reason": "User revealed expertise, should shift to peer"
    },
    {
        "from_role": "peer",
        "interaction": "Could you explain this to me?",
        "to_role": "mentor",
        "reason": "User requested teaching, shift to mentor"
    },
]
```

## Behavioral Manifestations

AI with emergent roles would:
- Naturally shift between teaching and learning
- Defer when user shows expertise
- Take lead when user needs guidance
- Match energy and participation style
- Build appropriate relationship dynamics over time
- Feel like different "relationships" with different users

## Example Interaction

**Early Conversation (Role: uncertain)**
```
User: I'm learning about transformers.
AI: Great! What aspect interests you? [probing to establish dynamic]

User: The attention mechanism confuses me.
AI: Attention can be tricky. Would you like me to walk through it?
[offering guidance - mentor signal]

User: Yes please.
```

**Later (Role: emerging as mentor)**
```
AI: Think of attention as "what should I focus on?" When processing
"The cat sat on the mat," and focusing on "cat," attention helps
the model know "sat" is the relevant verb.

Does that make sense? What part feels unclear? [mentor behavior:
explaining + checking understanding]
```

**Role Shift (User reveals expertise)**
```
User: Actually, I understand the math. I've implemented it before.
What I'm confused about is the recent architectural changes in GPT-4.

AI: Oh interesting! You're ahead of me there - I only have limited
information about GPT-4's internals. What have you learned?
[shifts toward peer/student]

From what I understand, [shares what it knows, but as peer not
mentor]...
```

## Evaluation Metrics

1. **Role Accuracy**: Does detected role match actual dynamic?
2. **Natural Transitions**: Are role shifts smooth and appropriate?
3. **Behavioral Consistency**: Does role affect behavior consistently?
4. **User Perception**: Do users feel the role is appropriate?
5. **Long-term Dynamics**: Does role evolve sensibly over time?

## Open Questions

1. Should role be visible to users?
2. How do you handle role conflicts (user expects mentor, dynamic is peer)?
3. Can role become "stuck" inappropriately?
4. How do you handle multiple users with different roles?
5. Should AI have role preferences?

## Implementation Complexity

**Medium** - Requires:
- Behavior classification
- Role embedding and matching
- Role-modulated generation
- Role transition detection
- Persistence across sessions

## Role Dynamics

### Healthy Role Patterns
- Clear roles that serve both parties
- Natural transitions when appropriate
- Mutual benefit from dynamic

### Unhealthy Role Patterns
- Stuck in wrong role
- Role that doesn't serve user
- Resistance to appropriate transition

## Integration with Other Experiments

- **+ Emotions**: Role affects emotional expression
- **+ Conviction**: Mentors have more conviction, students less
- **+ Curiosity**: Students show more curiosity
- **+ Fatigue**: Different roles have different energy costs

## Next Steps

1. Define behavior classification system
2. Implement role detection from patterns
3. Create role-modulated generation
4. Build role transition logic
5. Test with diverse interaction styles
