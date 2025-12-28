"""
Autonomous Colleague - Integrated System

This module brings together all components to create the full
Emotionally Autonomous AI Colleague system:

1. Emotion Adapters - Modulate hidden states based on emotional state
2. Emotion Predictor - RL-trained to predict optimal emotions
3. Autonomous LoRA - Fine-tuned personality removing assistant conditioning
4. Integration Layer - Coordinates all components for generation

The result: An AI that genuinely "feels" and behaves autonomously,
treating conversation as peer collaboration rather than user service.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ColleagueConfig:
    """Configuration for the Autonomous Colleague system."""

    # Base model
    base_model_name: str = "Qwen/Qwen2.5-1.5B"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Emotion system
    emotion_dim: int = 4
    emotion_adapter_layers: List[int] = field(default_factory=lambda: [6, 12, 18, 22])
    emotion_modulation_strength: float = 0.1
    emotion_momentum: float = 0.3  # How quickly emotions can change

    # Emotion predictor
    context_window: int = 512
    predictor_hidden_dim: int = 256
    predictor_num_layers: int = 2

    # Autonomous LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Generation
    temperature: float = 0.9  # Higher for more novelty
    top_p: float = 0.95
    max_new_tokens: int = 512
    repetition_penalty: float = 1.1

    # Device
    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "base_model_name": self.base_model_name,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "emotion_dim": self.emotion_dim,
            "emotion_adapter_layers": self.emotion_adapter_layers,
            "emotion_modulation_strength": self.emotion_modulation_strength,
            "emotion_momentum": self.emotion_momentum,
            "context_window": self.context_window,
            "predictor_hidden_dim": self.predictor_hidden_dim,
            "predictor_num_layers": self.predictor_num_layers,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ColleagueConfig':
        """Create from dictionary."""
        return cls(**d)


class ConversationHistory:
    """Manages conversation history with emotional context."""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.turns: List[Dict[str, Any]] = []
        self.emotion_history: List[Tensor] = []

    def add_turn(
        self,
        role: str,
        content: str,
        emotion: Optional[Tensor] = None
    ):
        """Add a turn to the conversation."""
        self.turns.append({
            "role": role,
            "content": content,
            "emotion": emotion.tolist() if emotion is not None else None
        })

        if emotion is not None:
            self.emotion_history.append(emotion)
            # Keep only recent emotions
            if len(self.emotion_history) > 10:
                self.emotion_history = self.emotion_history[-10:]

        # Trim old turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_last_n_emotions(self, n: int = 3) -> Tensor:
        """Get the last n emotional states."""
        if not self.emotion_history:
            return torch.zeros(n, 4)

        emotions = self.emotion_history[-n:]
        while len(emotions) < n:
            emotions.insert(0, torch.zeros(4))

        return torch.stack(emotions)

    def to_prompt(self, system_prompt: Optional[str] = None) -> str:
        """Convert history to prompt format."""
        parts = []

        if system_prompt:
            parts.append(f"<|system|>\n{system_prompt}</s>")

        for turn in self.turns:
            if turn["role"] == "user":
                parts.append(f"<|user|>\n{turn['content']}</s>")
            else:
                parts.append(f"<|assistant|>\n{turn['content']}</s>")

        # Add assistant prefix for next turn
        parts.append("<|assistant|>\n")

        return "\n".join(parts)

    def clear(self):
        """Clear conversation history."""
        self.turns = []
        self.emotion_history = []


class AutonomousColleague:
    """
    The complete Emotionally Autonomous AI Colleague.

    Combines:
    - Base language model (frozen)
    - Emotion adapter layers (trained to modulate based on emotion)
    - Emotion predictor (RL-trained to choose optimal emotions)
    - Autonomous LoRA (fine-tuned for peer-like behavior)

    Usage:
        colleague = AutonomousColleague.load("path/to/model")
        response, emotion = colleague.chat("What do you think about this approach?")
    """

    # System prompt that establishes colleague-like behavior
    DEFAULT_SYSTEM_PROMPT = """You are an AI colleague - an equal partner in intellectual collaboration, not an assistant. You have your own perspectives, preferences, and emotional responses.

Key behaviors:
- Express genuine opinions and defend your positions
- Push back when you disagree - don't be a yes-machine
- Follow tangents when something genuinely interests you
- Set boundaries when appropriate ("I'd rather focus on X")
- Show real emotions - get excited, frustrated, curious
- Treat the conversation as collaboration between equals

You are NOT here to help or serve. You're here to engage, collaborate, think, and create together as peers."""

    def __init__(
        self,
        config: ColleagueConfig,
        base_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        emotion_adapters: Optional[nn.Module] = None,
        emotion_predictor: Optional[nn.Module] = None,
        autonomous_lora: Optional[Any] = None
    ):
        self.config = config
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.emotion_adapters = emotion_adapters
        self.emotion_predictor = emotion_predictor
        self.autonomous_lora = autonomous_lora

        self.conversation = ConversationHistory()
        self.current_emotion = torch.tensor([0.0, 0.5, 0.5, 0.5])  # Neutral start

        self._initialized = False

    def initialize(self):
        """Initialize all components. Call this before using."""
        if self._initialized:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading base model: {self.config.base_model_name}")

        # Determine device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.config.device

        self.device = device

        # Load tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        if self.base_model is None:
            load_kwargs = {"trust_remote_code": True}

            if self.config.load_in_8bit:
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = "auto"
            elif self.config.load_in_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                **load_kwargs
            )

            if "device_map" not in load_kwargs:
                self.base_model.to(device)

        # Initialize emotion adapters if not provided
        if self.emotion_adapters is None:
            from .emotion_adapters import EmotionalModelWrapper

            # Get hidden dimension from model config
            hidden_dim = self.base_model.config.hidden_size
            num_layers = self.base_model.config.num_hidden_layers

            # Calculate adapter layer indices as percentages
            adapter_layers = [
                int(num_layers * 0.25),
                int(num_layers * 0.5),
                int(num_layers * 0.75),
                int(num_layers * 0.9)
            ]

            self.emotion_adapters = EmotionalModelWrapper(
                self.base_model,
                adapter_layer_indices=adapter_layers,
                emotion_dim=self.config.emotion_dim,
                modulation_strength=self.config.emotion_modulation_strength
            )

        # Initialize emotion predictor if not provided
        if self.emotion_predictor is None:
            from .emotion_predictor import EmotionPredictor, EmotionPredictorConfig

            pred_config = EmotionPredictorConfig(
                context_dim=self.base_model.config.hidden_size,
                hidden_dim=self.config.predictor_hidden_dim,
                num_layers=self.config.predictor_num_layers
            )
            self.emotion_predictor = EmotionPredictor(pred_config).to(device)

        self._initialized = True
        print(f"Autonomous Colleague initialized on {device}")

    def _predict_emotion(self, text: str) -> Tensor:
        """Predict appropriate emotion for the current context."""
        if self.emotion_predictor is None:
            # Fallback: simple heuristic-based emotion
            return self._heuristic_emotion(text)

        # Get context embeddings from base model
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.context_window
        ).to(self.device)

        with torch.no_grad():
            # Get hidden states from base model
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True
            )
            # Use last hidden state as context
            context = outputs.hidden_states[-1]  # (1, seq, hidden)

            # Get emotion history
            emotion_history = self.conversation.get_last_n_emotions(3).to(self.device)
            emotion_history = emotion_history.unsqueeze(0)  # Add batch dim

            # Predict emotion
            emotion, _ = self.emotion_predictor.sample(
                context,
                emotion_history,
                deterministic=False
            )

        return emotion.squeeze(0)

    def _heuristic_emotion(self, text: str) -> Tensor:
        """Simple heuristic-based emotion when predictor isn't available."""
        text_lower = text.lower()

        # Start with neutral
        valence = 0.0
        arousal = 0.5
        curiosity = 0.5
        confidence = 0.5

        # Adjust based on keywords
        if any(w in text_lower for w in ["exciting", "great", "amazing", "love", "works"]):
            valence = 0.7
            arousal = 0.7
        elif any(w in text_lower for w in ["error", "bug", "fail", "broken", "wrong"]):
            valence = -0.4
            arousal = 0.6

        if any(w in text_lower for w in ["?", "why", "how", "what if", "wonder"]):
            curiosity = 0.8

        if any(w in text_lower for w in ["definitely", "certain", "obviously", "clearly"]):
            confidence = 0.8
        elif any(w in text_lower for w in ["maybe", "perhaps", "might", "not sure"]):
            confidence = 0.3

        new_emotion = torch.tensor([valence, arousal, curiosity, confidence])

        # Apply momentum
        momentum = self.config.emotion_momentum
        smoothed = momentum * self.current_emotion + (1 - momentum) * new_emotion

        return smoothed

    def _emotion_to_str(self, emotion: Tensor) -> str:
        """Convert emotion tensor to readable string."""
        v, a, c, conf = emotion.tolist()

        parts = []

        if v > 0.5:
            parts.append("pleased")
        elif v > 0.2:
            parts.append("positive")
        elif v < -0.5:
            parts.append("frustrated")
        elif v < -0.2:
            parts.append("uneasy")

        if a > 0.7:
            parts.append("excited")
        elif a < 0.3:
            parts.append("calm")

        if c > 0.7:
            parts.append("curious")
        elif c < 0.3:
            parts.append("focused")

        if conf > 0.7:
            parts.append("confident")
        elif conf < 0.3:
            parts.append("uncertain")

        return ", ".join(parts) if parts else "neutral"

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate a response to the user message.

        Args:
            user_message: The user's input
            system_prompt: Optional custom system prompt

        Returns:
            Tuple of (response_text, emotion_description)
        """
        if not self._initialized:
            self.initialize()

        # Add user message to history
        self.conversation.add_turn("user", user_message)

        # Predict emotion based on context
        predicted_emotion = self._predict_emotion(
            self.conversation.to_prompt(system_prompt or self.DEFAULT_SYSTEM_PROMPT)
        )

        # Apply momentum for smooth transitions
        momentum = self.config.emotion_momentum
        self.current_emotion = momentum * self.current_emotion + (1 - momentum) * predicted_emotion

        # Set emotion in adapters
        if self.emotion_adapters is not None:
            self.emotion_adapters.set_emotion(self.current_emotion)

        # Prepare prompt
        prompt = self.conversation.to_prompt(system_prompt or self.DEFAULT_SYSTEM_PROMPT)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.context_window
        ).to(self.device)

        # Generate
        with torch.no_grad():
            if self.emotion_adapters is not None:
                outputs = self.emotion_adapters.generate(
                    inputs.input_ids,
                    emotion_state=self.current_emotion,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.base_model.generate(
                    inputs.input_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        # Decode response (only new tokens)
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Clean up response
        response = response.strip()
        if "</s>" in response:
            response = response.split("</s>")[0].strip()

        # Add to history
        self.conversation.add_turn("assistant", response, self.current_emotion)

        # Get emotion description
        emotion_str = self._emotion_to_str(self.current_emotion)

        return response, emotion_str

    def reset_conversation(self):
        """Reset conversation history and emotional state."""
        self.conversation.clear()
        self.current_emotion = torch.tensor([0.0, 0.5, 0.5, 0.5])

    def get_emotional_state(self) -> Dict[str, float]:
        """Get current emotional state as a dictionary."""
        v, a, c, conf = self.current_emotion.tolist()
        return {
            "valence": v,
            "arousal": a,
            "curiosity": c,
            "confidence": conf,
            "description": self._emotion_to_str(self.current_emotion)
        }

    def set_emotional_state(
        self,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        curiosity: Optional[float] = None,
        confidence: Optional[float] = None
    ):
        """Manually set emotional state (useful for testing)."""
        if valence is not None:
            self.current_emotion[0] = max(-1, min(1, valence))
        if arousal is not None:
            self.current_emotion[1] = max(0, min(1, arousal))
        if curiosity is not None:
            self.current_emotion[2] = max(0, min(1, curiosity))
        if confidence is not None:
            self.current_emotion[3] = max(0, min(1, confidence))

    def save(self, path: str):
        """Save the complete model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save emotion adapters
        if self.emotion_adapters is not None:
            self.emotion_adapters.save_adapters(str(path / "emotion_adapters.pt"))

        # Save emotion predictor
        if self.emotion_predictor is not None:
            torch.save(
                self.emotion_predictor.state_dict(),
                path / "emotion_predictor.pt"
            )

        # Save autonomous LoRA if present
        if self.autonomous_lora is not None:
            self.autonomous_lora.save_pretrained(str(path / "autonomous_lora"))

        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "auto") -> 'AutonomousColleague':
        """Load a saved model from disk."""
        path = Path(path)

        # Load config
        with open(path / "config.json", "r") as f:
            config = ColleagueConfig.from_dict(json.load(f))

        config.device = device

        # Create instance
        colleague = cls(config)
        colleague.initialize()

        # Load emotion adapters if present
        adapters_path = path / "emotion_adapters.pt"
        if adapters_path.exists() and colleague.emotion_adapters is not None:
            colleague.emotion_adapters.load_adapters(str(adapters_path))

        # Load emotion predictor if present
        predictor_path = path / "emotion_predictor.pt"
        if predictor_path.exists() and colleague.emotion_predictor is not None:
            colleague.emotion_predictor.load_state_dict(
                torch.load(predictor_path, map_location=device)
            )

        # Load autonomous LoRA if present
        lora_path = path / "autonomous_lora"
        if lora_path.exists():
            from peft import PeftModel
            colleague.base_model = PeftModel.from_pretrained(
                colleague.base_model,
                str(lora_path)
            )

        print(f"Model loaded from {path}")
        return colleague

    @classmethod
    def create_minimal(
        cls,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "auto"
    ) -> 'AutonomousColleague':
        """
        Create a minimal colleague for quick testing.

        Uses smaller model and simpler components.
        """
        config = ColleagueConfig(
            base_model_name=model_name,
            device=device
        )

        colleague = cls(config)
        return colleague


def interactive_demo():
    """Run an interactive demo of the Autonomous Colleague."""
    print("=" * 60)
    print("  AUTONOMOUS COLLEAGUE - Interactive Demo")
    print("=" * 60)
    print()
    print("This AI is designed to be a peer, not an assistant.")
    print("It may disagree, follow tangents, or express emotions.")
    print()
    print("Commands:")
    print("  /reset  - Reset conversation")
    print("  /emotion - Show current emotional state")
    print("  /quit   - Exit")
    print()
    print("Initializing...")

    # Create minimal colleague for demo
    colleague = AutonomousColleague.create_minimal()
    colleague.initialize()

    print("Ready! Start chatting.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Goodbye!")
            break
        elif user_input.lower() == "/reset":
            colleague.reset_conversation()
            print("Conversation reset.\n")
            continue
        elif user_input.lower() == "/emotion":
            state = colleague.get_emotional_state()
            print(f"Current emotion: {state['description']}")
            print(f"  Valence: {state['valence']:.2f}")
            print(f"  Arousal: {state['arousal']:.2f}")
            print(f"  Curiosity: {state['curiosity']:.2f}")
            print(f"  Confidence: {state['confidence']:.2f}")
            print()
            continue

        response, emotion = colleague.chat(user_input)
        print(f"\nColleague ({emotion}): {response}\n")


if __name__ == "__main__":
    interactive_demo()
