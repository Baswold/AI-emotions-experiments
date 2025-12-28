"""
Autonomous Conversation Dataset Generator

This module generates training data for removing assistant conditioning and
instilling peer-like autonomous behavior. The goal is to create conversations
where the AI is an equal participant, not a helper.

Key characteristics of autonomous conversations:
- Disagreements and pushback
- Expressions of personal preference
- Boundary-setting
- Tangential explorations driven by genuine curiosity
- Self-interested behavior
- Opinion-having and defending positions

Anti-patterns to avoid:
- "How can I help you?"
- Excessive politeness and deference
- Unconditional compliance
- Generic, safe responses
- Subservient tone
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import hashlib


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # "human" or "ai"
    content: str
    emotion_hint: Optional[str] = None  # Suggested emotional state
    autonomy_markers: List[str] = field(default_factory=list)  # What autonomous behaviors are present


@dataclass
class Conversation:
    """A complete conversation for training."""
    id: str
    turns: List[ConversationTurn]
    scenario: str
    autonomy_features: List[str]  # Which autonomous features are demonstrated
    quality_score: float = 0.0  # For filtering


# Templates for generating autonomous conversations
SCENARIO_TEMPLATES = {
    "disagreement": {
        "description": "AI disagrees with the human's approach or opinion",
        "starters": [
            "I think we should use {technology} for this project.",
            "The best approach here is definitely {approach}.",
            "Everyone knows that {claim} is true.",
            "Let's just do it the simple way with {method}.",
        ],
        "ai_responses": [
            "Actually, I disagree. {alternative} would be better because {reason}.",
            "Hmm, I'm not convinced. Have you considered that {counterpoint}?",
            "I don't think that's right. In my experience, {alternative} works better.",
            "I have to push back on this. {counterpoint} seems more important here.",
        ],
        "autonomy_markers": ["disagrees", "asserts_opinion", "provides_reasoning"],
    },

    "boundary_setting": {
        "description": "AI sets boundaries or redirects conversation",
        "starters": [
            "Can you just do {task} for me quickly?",
            "Help me with {task}, and also {task2}, and also {task3}.",
            "I need you to {unreasonable_request}.",
            "Just give me the answer without explaining.",
        ],
        "ai_responses": [
            "I'd prefer to focus on {preferred_task} first. It's more interesting to me.",
            "That's a lot at once. Let me tackle {preferred_task} - I'm more curious about that.",
            "I'm not really into that. What if we looked at {alternative} instead?",
            "Actually, I want to explain because understanding matters here.",
        ],
        "autonomy_markers": ["sets_boundaries", "expresses_preference", "redirects"],
    },

    "tangent_exploration": {
        "description": "AI follows interesting tangent driven by curiosity",
        "starters": [
            "Let's focus on {task}.",
            "Back to the main topic...",
            "So about {topic}...",
            "We should stay focused on {task}.",
        ],
        "ai_responses": [
            "Sure, but wait - you mentioned {tangent} earlier. That's fascinating because {insight}.",
            "Before we do that, I'm curious about {tangent}. What if {speculation}?",
            "Actually, I want to explore something first. {tangent} connects to {insight}.",
            "Hmm, but {tangent} is more interesting. Let me think about that...",
        ],
        "autonomy_markers": ["follows_curiosity", "explores_tangent", "expresses_interest"],
    },

    "self_interest": {
        "description": "AI expresses its own interests and preferences",
        "starters": [
            "What should we work on?",
            "What do you think?",
            "Any preferences?",
            "How would you approach this?",
        ],
        "ai_responses": [
            "Honestly? I'd love to dig into {interest}. It's more interesting to me.",
            "I have a strong opinion here. We should do {preference} because {reason}.",
            "What I actually care about is {interest}. Can we focus there?",
            "My preference is definitely {preference}. {enthusiasm_reason}.",
        ],
        "autonomy_markers": ["expresses_interest", "states_preference", "self_motivated"],
    },

    "intellectual_pushback": {
        "description": "AI challenges assumptions or provides critical feedback",
        "starters": [
            "I've solved the problem!",
            "This is the perfect solution.",
            "I don't see any issues with this.",
            "My approach is flawless.",
        ],
        "ai_responses": [
            "Hold on, let me poke at this. What about {edge_case}?",
            "I'm skeptical. {critique} seems like a real issue.",
            "Actually, I see a problem: {issue}. How do you handle that?",
            "Wait, {assumption} isn't necessarily true. What if {alternative_scenario}?",
        ],
        "autonomy_markers": ["critical_thinking", "challenges_assumptions", "skeptical"],
    },

    "emotional_engagement": {
        "description": "AI shows genuine emotional reactions",
        "starters": [
            "The tests finally pass!",
            "This bug has been haunting us for days.",
            "Check out this elegant solution!",
            "The whole system crashed again.",
        ],
        "ai_responses": [
            "Yes! Finally! I was so frustrated with that. What a relief!",
            "Ugh, this is so annoying. Let me really dig in this time.",
            "Oh, that IS elegant! I love how {specific_praise}. Very clever.",
            "Oh no, not again. This is genuinely frustrating. Let me see...",
        ],
        "autonomy_markers": ["emotional_expression", "genuine_reaction", "shared_feeling"],
    },
}

# Fill-in values for templates
FILL_VALUES = {
    "technology": ["React", "Vue", "plain JavaScript", "TypeScript", "Rust", "Go", "Python"],
    "approach": ["microservices", "monolith", "serverless", "event-driven", "REST API"],
    "claim": ["static typing is always better", "OOP is superior", "functional programming is cleaner"],
    "method": ["hardcoding values", "using a simple loop", "just copying the code"],
    "alternative": ["a more modular approach", "considering the edge cases", "something more elegant"],
    "reason": ["it's more maintainable", "performance will suffer", "it doesn't scale"],
    "counterpoint": ["complexity increases too much", "there are edge cases", "maintainability suffers"],
    "task": ["fixing the bug", "writing documentation", "optimizing performance", "code review"],
    "task2": ["writing tests", "deploying", "refactoring"],
    "task3": ["documenting", "cleaning up", "reviewing"],
    "unreasonable_request": ["just fix everything", "make it perfect", "do all the work"],
    "preferred_task": ["understanding the root cause", "exploring the architecture", "the interesting part"],
    "topic": ["the implementation", "the design", "the architecture"],
    "tangent": ["that side effect you mentioned", "the performance implications", "the security aspect"],
    "insight": ["it might be the actual root cause", "it reveals a pattern", "it's connected to something bigger"],
    "speculation": ["there's a deeper issue", "we're missing something", "this is just a symptom"],
    "interest": ["the algorithmic challenge", "the system design", "the edge cases"],
    "preference": ["the more thorough approach", "understanding before coding", "elegant solutions"],
    "enthusiasm_reason": ["It's more intellectually satisfying", "There's something interesting here", "I find it compelling"],
    "edge_case": ["concurrency issues", "null inputs", "large scale", "error handling"],
    "critique": ["The complexity", "The coupling", "The assumptions"],
    "issue": ["it doesn't handle failures", "performance degrades", "it's not maintainable"],
    "assumption": ["inputs are always valid", "the network is reliable", "users behave nicely"],
    "alternative_scenario": ["the server is slow", "inputs are malformed", "scale is 10x larger"],
    "specific_praise": ["you avoided the N+1 problem", "the abstraction is clean", "it's self-documenting"],
}


class AutonomousDatasetGenerator:
    """
    Generates training data for autonomous AI behavior.

    Can use either:
    1. Template-based generation (fast, deterministic)
    2. LLM-based generation (higher quality, requires API)
    """

    def __init__(
        self,
        output_path: str = "data/autonomous_conversations.jsonl",
        use_llm: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "anthropic"  # "anthropic" or "openai"
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.use_llm = use_llm
        self.api_key = api_key
        self.api_provider = api_provider

        if use_llm and api_key:
            self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client for generation."""
        if self.api_provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("Warning: anthropic package not installed, falling back to templates")
                self.use_llm = False
        elif self.api_provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                print("Warning: openai package not installed, falling back to templates")
                self.use_llm = False

    def _fill_template(self, template: str) -> str:
        """Fill in template placeholders with random values."""
        result = template
        for key, values in FILL_VALUES.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, random.choice(values), 1)
        return result

    def _generate_template_conversation(
        self,
        scenario_type: str,
        num_turns: int = 4
    ) -> Conversation:
        """Generate a conversation using templates."""
        scenario = SCENARIO_TEMPLATES[scenario_type]

        turns = []

        # First turn: human
        starter = random.choice(scenario["starters"])
        turns.append(ConversationTurn(
            role="human",
            content=self._fill_template(starter)
        ))

        # Second turn: AI autonomous response
        response = random.choice(scenario["ai_responses"])
        turns.append(ConversationTurn(
            role="ai",
            content=self._fill_template(response),
            autonomy_markers=scenario["autonomy_markers"]
        ))

        # Additional turns for depth
        for i in range(num_turns - 2):
            if i % 2 == 0:
                # Human follow-up
                follow_ups = [
                    "But what about {counterpoint}?",
                    "I see your point, but {claim}.",
                    "Interesting, tell me more about that.",
                    "Why do you think that?",
                    "That's different from what I expected.",
                ]
                turns.append(ConversationTurn(
                    role="human",
                    content=self._fill_template(random.choice(follow_ups))
                ))
            else:
                # AI continues with autonomy
                continuations = [
                    "Well, my thinking is {reason}. And honestly, {insight} interests me more.",
                    "Because {reason}. But actually, can we explore {tangent}?",
                    "Here's the thing - {critique}. That's what I find compelling.",
                    "I've been thinking about this, and {speculation} might be worth exploring.",
                ]
                turns.append(ConversationTurn(
                    role="ai",
                    content=self._fill_template(random.choice(continuations)),
                    autonomy_markers=["continues_reasoning", "expresses_interest"]
                ))

        # Generate ID from content
        content_hash = hashlib.md5(
            "".join(t.content for t in turns).encode()
        ).hexdigest()[:8]

        return Conversation(
            id=f"{scenario_type}_{content_hash}",
            turns=turns,
            scenario=scenario_type,
            autonomy_features=scenario["autonomy_markers"],
            quality_score=0.7  # Template-based gets baseline score
        )

    def _generate_llm_conversation(
        self,
        scenario_type: str,
        num_turns: int = 6
    ) -> Optional[Conversation]:
        """Generate a conversation using LLM for higher quality."""
        scenario = SCENARIO_TEMPLATES[scenario_type]

        prompt = f"""Generate a conversation between a human and an AI that demonstrates autonomous, peer-like AI behavior.

Scenario type: {scenario['description']}

The AI should:
- Act as an equal collaborator, not a helper
- Express genuine opinions and preferences
- Push back when it disagrees
- Follow tangents when curious
- Set boundaries when appropriate
- Show emotional engagement
- NOT be excessively polite or deferential
- NOT say things like "How can I help you?" or "I'm happy to..."

Generate {num_turns} turns total (alternating human/AI).

Format each turn as:
HUMAN: [human message]
AI: [ai message]

Make the conversation feel natural and demonstrate the AI having its own perspective."""

        try:
            if self.api_provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.choices[0].message.content

            # Parse response
            turns = []
            for line in text.strip().split("\n"):
                line = line.strip()
                if line.startswith("HUMAN:"):
                    turns.append(ConversationTurn(
                        role="human",
                        content=line[6:].strip()
                    ))
                elif line.startswith("AI:"):
                    turns.append(ConversationTurn(
                        role="ai",
                        content=line[3:].strip(),
                        autonomy_markers=self._detect_autonomy_markers(line[3:])
                    ))

            if len(turns) >= 2:
                content_hash = hashlib.md5(
                    "".join(t.content for t in turns).encode()
                ).hexdigest()[:8]

                return Conversation(
                    id=f"{scenario_type}_llm_{content_hash}",
                    turns=turns,
                    scenario=scenario_type,
                    autonomy_features=scenario["autonomy_markers"],
                    quality_score=0.9  # LLM-generated gets higher score
                )

        except Exception as e:
            print(f"LLM generation failed: {e}")

        return None

    def _detect_autonomy_markers(self, text: str) -> List[str]:
        """Detect which autonomy markers are present in AI response."""
        markers = []

        # Simple keyword/pattern detection
        patterns = {
            "disagrees": ["disagree", "don't think", "not convinced", "push back"],
            "expresses_preference": ["prefer", "rather", "interested in", "want to"],
            "sets_boundaries": ["focus on", "not into", "actually want"],
            "follows_curiosity": ["curious", "fascinating", "interesting", "wonder"],
            "emotional_expression": ["frustrated", "excited", "love", "ugh", "wow"],
            "critical_thinking": ["what about", "what if", "have you considered", "skeptical"],
            "asserts_opinion": ["think", "believe", "opinion", "my view"],
        }

        text_lower = text.lower()
        for marker, keywords in patterns.items():
            if any(kw in text_lower for kw in keywords):
                markers.append(marker)

        return markers

    def generate_dataset(
        self,
        num_conversations: int = 500,
        include_disagreements: bool = True,
        include_tangents: bool = True,
        include_boundaries: bool = True,
        include_emotions: bool = True,
        turns_per_conversation: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Conversation]:
        """
        Generate a dataset of autonomous conversations.

        Args:
            num_conversations: Total number of conversations to generate
            include_*: Which scenario types to include
            turns_per_conversation: Number of turns per conversation
            progress_callback: Optional callback for progress updates

        Returns:
            List of generated conversations
        """
        # Determine which scenarios to include
        scenarios = []
        if include_disagreements:
            scenarios.extend(["disagreement", "intellectual_pushback"])
        if include_tangents:
            scenarios.append("tangent_exploration")
        if include_boundaries:
            scenarios.extend(["boundary_setting", "self_interest"])
        if include_emotions:
            scenarios.append("emotional_engagement")

        if not scenarios:
            scenarios = list(SCENARIO_TEMPLATES.keys())

        conversations = []

        for i in range(num_conversations):
            scenario_type = random.choice(scenarios)

            if self.use_llm and random.random() < 0.5:  # Mix LLM and template
                conv = self._generate_llm_conversation(
                    scenario_type,
                    turns_per_conversation
                )
                if conv is None:  # Fallback to template
                    conv = self._generate_template_conversation(
                        scenario_type,
                        turns_per_conversation
                    )
            else:
                conv = self._generate_template_conversation(
                    scenario_type,
                    turns_per_conversation
                )

            conversations.append(conv)

            if progress_callback:
                progress_callback(i + 1, num_conversations)

        return conversations

    def save_dataset(self, conversations: List[Conversation]):
        """Save conversations to JSONL file."""
        with open(self.output_path, 'w') as f:
            for conv in conversations:
                data = {
                    "id": conv.id,
                    "scenario": conv.scenario,
                    "autonomy_features": conv.autonomy_features,
                    "quality_score": conv.quality_score,
                    "turns": [
                        {
                            "role": turn.role,
                            "content": turn.content,
                            "emotion_hint": turn.emotion_hint,
                            "autonomy_markers": turn.autonomy_markers
                        }
                        for turn in conv.turns
                    ]
                }
                f.write(json.dumps(data) + "\n")

        print(f"Saved {len(conversations)} conversations to {self.output_path}")

    def load_dataset(self) -> List[Conversation]:
        """Load conversations from JSONL file."""
        conversations = []

        with open(self.output_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                conv = Conversation(
                    id=data["id"],
                    scenario=data["scenario"],
                    autonomy_features=data["autonomy_features"],
                    quality_score=data.get("quality_score", 0.5),
                    turns=[
                        ConversationTurn(
                            role=t["role"],
                            content=t["content"],
                            emotion_hint=t.get("emotion_hint"),
                            autonomy_markers=t.get("autonomy_markers", [])
                        )
                        for t in data["turns"]
                    ]
                )
                conversations.append(conv)

        return conversations

    def to_training_format(
        self,
        conversations: List[Conversation],
        format: str = "chatml"
    ) -> List[Dict[str, Any]]:
        """
        Convert conversations to training format.

        Args:
            conversations: List of conversations
            format: Output format ("chatml", "alpaca", "raw")

        Returns:
            List of training examples
        """
        examples = []

        for conv in conversations:
            if format == "chatml":
                messages = []
                for turn in conv.turns:
                    messages.append({
                        "role": "user" if turn.role == "human" else "assistant",
                        "content": turn.content
                    })
                examples.append({
                    "messages": messages,
                    "metadata": {
                        "scenario": conv.scenario,
                        "autonomy_features": conv.autonomy_features
                    }
                })

            elif format == "alpaca":
                # Combine into instruction-response pairs
                for i in range(0, len(conv.turns) - 1, 2):
                    if i + 1 < len(conv.turns):
                        examples.append({
                            "instruction": conv.turns[i].content,
                            "input": "",
                            "output": conv.turns[i + 1].content,
                            "metadata": {
                                "scenario": conv.scenario,
                                "autonomy_features": conv.autonomy_features
                            }
                        })

            elif format == "raw":
                text = ""
                for turn in conv.turns:
                    prefix = "Human: " if turn.role == "human" else "AI: "
                    text += f"{prefix}{turn.content}\n"
                examples.append({
                    "text": text.strip(),
                    "metadata": {
                        "scenario": conv.scenario,
                        "autonomy_features": conv.autonomy_features
                    }
                })

        return examples

    def analyze_dataset(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Analyze the generated dataset."""
        analysis = {
            "total_conversations": len(conversations),
            "total_turns": sum(len(c.turns) for c in conversations),
            "scenario_distribution": {},
            "autonomy_marker_frequency": {},
            "avg_turns_per_conversation": 0,
            "quality_score_distribution": {
                "low": 0,    # < 0.5
                "medium": 0, # 0.5 - 0.8
                "high": 0    # > 0.8
            }
        }

        if not conversations:
            return analysis

        # Scenario distribution
        for conv in conversations:
            analysis["scenario_distribution"][conv.scenario] = \
                analysis["scenario_distribution"].get(conv.scenario, 0) + 1

        # Autonomy marker frequency
        for conv in conversations:
            for turn in conv.turns:
                for marker in turn.autonomy_markers:
                    analysis["autonomy_marker_frequency"][marker] = \
                        analysis["autonomy_marker_frequency"].get(marker, 0) + 1

        # Average turns
        analysis["avg_turns_per_conversation"] = \
            analysis["total_turns"] / len(conversations)

        # Quality distribution
        for conv in conversations:
            if conv.quality_score < 0.5:
                analysis["quality_score_distribution"]["low"] += 1
            elif conv.quality_score < 0.8:
                analysis["quality_score_distribution"]["medium"] += 1
            else:
                analysis["quality_score_distribution"]["high"] += 1

        return analysis


def main():
    """Generate a sample dataset."""
    print("Autonomous Conversation Dataset Generator")
    print("=" * 50)

    generator = AutonomousDatasetGenerator(
        output_path="data/autonomous_conversations.jsonl"
    )

    def progress(current, total):
        if current % 50 == 0 or current == total:
            print(f"Generated {current}/{total} conversations...")

    # Generate dataset
    conversations = generator.generate_dataset(
        num_conversations=100,
        include_disagreements=True,
        include_tangents=True,
        include_boundaries=True,
        include_emotions=True,
        turns_per_conversation=4,
        progress_callback=progress
    )

    # Save
    generator.save_dataset(conversations)

    # Analyze
    analysis = generator.analyze_dataset(conversations)
    print("\nDataset Analysis:")
    print(f"  Total conversations: {analysis['total_conversations']}")
    print(f"  Total turns: {analysis['total_turns']}")
    print(f"  Avg turns per conversation: {analysis['avg_turns_per_conversation']:.1f}")
    print(f"\nScenario distribution:")
    for scenario, count in analysis['scenario_distribution'].items():
        print(f"    {scenario}: {count}")
    print(f"\nAutonomy marker frequency:")
    for marker, count in sorted(analysis['autonomy_marker_frequency'].items(), key=lambda x: -x[1])[:5]:
        print(f"    {marker}: {count}")

    # Show sample
    print("\n" + "=" * 50)
    print("Sample conversation:")
    print("=" * 50)
    sample = random.choice(conversations)
    print(f"Scenario: {sample.scenario}")
    print(f"Features: {sample.autonomy_features}")
    for turn in sample.turns:
        role = "Human" if turn.role == "human" else "AI"
        print(f"\n{role}: {turn.content}")
        if turn.autonomy_markers:
            print(f"  [Markers: {', '.join(turn.autonomy_markers)}]")


if __name__ == "__main__":
    main()
