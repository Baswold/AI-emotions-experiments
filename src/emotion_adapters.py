"""
Emotion Adapter Layers for Computational Emotional Modulation

This module implements the core innovation: adapter layers that directly modulate
the hidden states of a language model based on a continuous emotional state vector.

Key insight: Emotions in humans are computational shortcuts - fear narrows attention,
curiosity broadens it. They're not just labels, they mechanically change how we
process information. This module replicates that for AI.

Architecture:
    h' = h * (1 + α * scale(emotion)) + bias(emotion)

Where:
    - h: original hidden states
    - emotion: 4D vector (valence, arousal, curiosity, confidence)
    - scale/bias: learned projections from emotion space to hidden dimension
    - α: modulation strength (hyperparameter)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class EmotionState:
    """
    Continuous emotional state vector in 4D space.

    Unlike discrete emotion labels (happy, sad, angry), this allows for
    nuanced combinations like "slightly anxious and very curious" through
    continuous values.

    Attributes:
        valence: -1 to 1 (negative/frustrated ↔ positive/pleased)
        arousal: 0 to 1 (calm/relaxed ↔ excited/energized)
        curiosity: 0 to 1 (focused/task-bound ↔ exploratory/tangent-seeking)
        confidence: 0 to 1 (uncertain/tentative ↔ certain/assertive)
    """
    valence: float = 0.0      # -1 to 1
    arousal: float = 0.5      # 0 to 1
    curiosity: float = 0.5    # 0 to 1
    confidence: float = 0.5   # 0 to 1

    # Momentum for smooth transitions
    _previous: Optional['EmotionState'] = field(default=None, repr=False)
    momentum: float = field(default=0.3, repr=False)  # How much previous state affects current

    def __post_init__(self):
        """Validate and clamp values to valid ranges."""
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.curiosity = max(0.0, min(1.0, self.curiosity))
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_tensor(self, device: str = "cpu") -> Tensor:
        """Convert to PyTorch tensor of shape (4,)."""
        return torch.tensor(
            [self.valence, self.arousal, self.curiosity, self.confidence],
            dtype=torch.float32,
            device=device
        )

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> 'EmotionState':
        """Create EmotionState from tensor."""
        values = tensor.detach().cpu().tolist()
        return cls(
            valence=values[0],
            arousal=values[1],
            curiosity=values[2],
            confidence=values[3]
        )

    def with_momentum(self, new_state: 'EmotionState') -> 'EmotionState':
        """
        Transition to new state with momentum (emotions can't flip instantly).

        This creates authentic emotional arcs rather than jarring transitions.
        """
        return EmotionState(
            valence=self.momentum * self.valence + (1 - self.momentum) * new_state.valence,
            arousal=self.momentum * self.arousal + (1 - self.momentum) * new_state.arousal,
            curiosity=self.momentum * self.curiosity + (1 - self.momentum) * new_state.curiosity,
            confidence=self.momentum * self.confidence + (1 - self.momentum) * new_state.confidence,
            _previous=self,
            momentum=self.momentum
        )

    def distance(self, other: 'EmotionState') -> float:
        """Euclidean distance between emotional states."""
        return math.sqrt(
            (self.valence - other.valence) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.curiosity - other.curiosity) ** 2 +
            (self.confidence - other.confidence) ** 2
        )

    def __str__(self) -> str:
        """Human-readable emotion description."""
        parts = []

        # Valence
        if self.valence > 0.5:
            parts.append("pleased")
        elif self.valence > 0.2:
            parts.append("positive")
        elif self.valence < -0.5:
            parts.append("frustrated")
        elif self.valence < -0.2:
            parts.append("negative")

        # Arousal
        if self.arousal > 0.7:
            parts.append("excited")
        elif self.arousal < 0.3:
            parts.append("calm")

        # Curiosity
        if self.curiosity > 0.7:
            parts.append("curious")
        elif self.curiosity < 0.3:
            parts.append("focused")

        # Confidence
        if self.confidence > 0.7:
            parts.append("confident")
        elif self.confidence < 0.3:
            parts.append("uncertain")

        return ", ".join(parts) if parts else "neutral"


# Preset emotional states for common situations
EMOTION_PRESETS = {
    "neutral": EmotionState(0.0, 0.5, 0.5, 0.5),
    "curious": EmotionState(0.3, 0.6, 0.9, 0.5),
    "excited": EmotionState(0.8, 0.9, 0.7, 0.7),
    "frustrated": EmotionState(-0.6, 0.7, 0.2, 0.4),
    "calm_confident": EmotionState(0.2, 0.3, 0.4, 0.8),
    "anxious": EmotionState(-0.3, 0.8, 0.3, 0.2),
    "deeply_focused": EmotionState(0.1, 0.5, 0.1, 0.7),
    "playful": EmotionState(0.7, 0.7, 0.8, 0.6),
    "skeptical": EmotionState(-0.2, 0.4, 0.6, 0.3),
    "inspired": EmotionState(0.9, 0.8, 0.9, 0.8),
}


class EmotionAdapter(nn.Module):
    """
    Adapter layer that modulates hidden states based on emotional state.

    The key formula:
        h' = h * (1 + α * scale(emotion)) + bias(emotion)

    This is NOT just adding information to the representation - it's
    actually changing HOW the model processes information, similar to
    how human emotions affect cognition.

    Args:
        hidden_dim: Dimension of the hidden states to modulate
        emotion_dim: Dimension of the emotion vector (default 4)
        intermediate_dim: Size of intermediate projection (default hidden_dim // 4)
        modulation_strength: α parameter controlling modulation intensity
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        hidden_dim: int,
        emotion_dim: int = 4,
        intermediate_dim: Optional[int] = None,
        modulation_strength: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.emotion_dim = emotion_dim
        self.intermediate_dim = intermediate_dim or (hidden_dim // 4)
        self.modulation_strength = modulation_strength

        # Project emotion to intermediate space
        self.emotion_proj = nn.Sequential(
            nn.Linear(emotion_dim, self.intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.intermediate_dim, self.intermediate_dim),
            nn.GELU(),
        )

        # Generate scale factors (multiplicative modulation)
        self.scale_proj = nn.Sequential(
            nn.Linear(self.intermediate_dim, hidden_dim),
            nn.Tanh(),  # Bounded to [-1, 1] for stability
        )

        # Generate bias (additive modulation)
        self.bias_proj = nn.Sequential(
            nn.Linear(self.intermediate_dim, hidden_dim),
            nn.Tanh(),
        )

        # Learnable gate to control how much modulation to apply
        self.gate = nn.Parameter(torch.ones(1) * 0.5)

        # Initialize weights for minimal initial impact
        self._init_weights()

    def _init_weights(self):
        """Initialize to near-identity transformation initially."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: Tensor,
        emotion_state: Tensor
    ) -> Tensor:
        """
        Apply emotional modulation to hidden states.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            emotion_state: (batch, 4) or (4,) emotion vector

        Returns:
            Modulated hidden states of same shape
        """
        # Ensure emotion_state has batch dimension
        if emotion_state.dim() == 1:
            emotion_state = emotion_state.unsqueeze(0)

        # Expand emotion to match batch size if needed
        if emotion_state.size(0) == 1 and hidden_states.size(0) > 1:
            emotion_state = emotion_state.expand(hidden_states.size(0), -1)

        # Project emotion to feature space
        emotion_features = self.emotion_proj(emotion_state)  # (batch, intermediate)

        # Generate scale and bias
        scale = self.scale_proj(emotion_features)  # (batch, hidden_dim)
        bias = self.bias_proj(emotion_features)    # (batch, hidden_dim)

        # Expand for sequence dimension: (batch, 1, hidden_dim)
        scale = scale.unsqueeze(1)
        bias = bias.unsqueeze(1)

        # Apply modulation with gating
        gate = torch.sigmoid(self.gate)
        modulated = hidden_states * (1.0 + gate * self.modulation_strength * scale)
        modulated = modulated + gate * self.modulation_strength * bias

        return modulated

    def get_modulation_stats(
        self,
        emotion_state: Tensor
    ) -> Dict[str, float]:
        """Get statistics about the modulation for analysis."""
        if emotion_state.dim() == 1:
            emotion_state = emotion_state.unsqueeze(0)

        with torch.no_grad():
            emotion_features = self.emotion_proj(emotion_state)
            scale = self.scale_proj(emotion_features)
            bias = self.bias_proj(emotion_features)

            return {
                "scale_mean": scale.mean().item(),
                "scale_std": scale.std().item(),
                "scale_max": scale.abs().max().item(),
                "bias_mean": bias.mean().item(),
                "bias_std": bias.std().item(),
                "bias_max": bias.abs().max().item(),
                "gate_value": torch.sigmoid(self.gate).item(),
            }


class EmotionalModelWrapper(nn.Module):
    """
    Wraps a base language model with emotion adapters at multiple layers.

    Adapters are injected at strategic depths:
    - Early layers (~25%): Affect low-level feature extraction
    - Middle layers (~50%): Change reasoning patterns
    - Late layers (~75%, 90%): Modify output selection

    This creates deep computational change throughout the network.

    Args:
        base_model: The underlying language model
        adapter_layer_indices: Which layers to inject adapters at
        emotion_dim: Dimension of emotion vector
        modulation_strength: How strongly emotions affect computation
    """

    def __init__(
        self,
        base_model: nn.Module,
        adapter_layer_indices: Optional[List[int]] = None,
        emotion_dim: int = 4,
        modulation_strength: float = 0.1,
        freeze_base: bool = True
    ):
        super().__init__()

        self.base_model = base_model
        self.emotion_dim = emotion_dim

        # Freeze base model parameters
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Detect model architecture and hidden dimension
        self.hidden_dim = self._detect_hidden_dim()
        self.num_layers = self._detect_num_layers()

        # Determine adapter injection points
        if adapter_layer_indices is None:
            # Default: inject at 25%, 50%, 75%, 90% through the network
            adapter_layer_indices = [
                int(self.num_layers * 0.25),
                int(self.num_layers * 0.5),
                int(self.num_layers * 0.75),
                int(self.num_layers * 0.9),
            ]

        self.adapter_layer_indices = set(adapter_layer_indices)

        # Create adapters for each injection point
        self.emotion_adapters = nn.ModuleDict({
            str(idx): EmotionAdapter(
                hidden_dim=self.hidden_dim,
                emotion_dim=emotion_dim,
                modulation_strength=modulation_strength
            )
            for idx in adapter_layer_indices
        })

        # Current emotional state
        self.current_emotion = EMOTION_PRESETS["neutral"].to_tensor()

        # Register hooks for adapter injection
        self._register_hooks()

    def _detect_hidden_dim(self) -> int:
        """Detect hidden dimension from model config."""
        config = getattr(self.base_model, 'config', None)
        if config:
            for attr in ['hidden_size', 'd_model', 'n_embd']:
                if hasattr(config, attr):
                    return getattr(config, attr)

        # Fallback: try to infer from model structure
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                return module.out_features

        raise ValueError("Could not detect hidden dimension")

    def _detect_num_layers(self) -> int:
        """Detect number of transformer layers."""
        config = getattr(self.base_model, 'config', None)
        if config:
            for attr in ['num_hidden_layers', 'n_layer', 'num_layers']:
                if hasattr(config, attr):
                    return getattr(config, attr)

        # Fallback: count layers
        count = 0
        for name, _ in self.base_model.named_modules():
            if 'layer' in name.lower() or 'block' in name.lower():
                count += 1

        return max(count // 2, 1)  # Rough estimate

    def _register_hooks(self):
        """Register forward hooks to inject emotion adapters."""
        self._hooks = []
        layer_idx = 0

        def make_hook(adapter_key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    rest = output[1:]
                else:
                    hidden_states = output
                    rest = None

                # Apply emotional modulation
                emotion = self.current_emotion.to(hidden_states.device)
                modulated = self.emotion_adapters[adapter_key](hidden_states, emotion)

                if rest is not None:
                    return (modulated,) + rest
                return modulated

            return hook

        # Find transformer layers and attach hooks
        for name, module in self.base_model.named_modules():
            # Common layer naming patterns
            if any(pattern in name.lower() for pattern in ['layers.', 'h.', 'blocks.']):
                if layer_idx in self.adapter_layer_indices:
                    hook = module.register_forward_hook(make_hook(str(layer_idx)))
                    self._hooks.append(hook)
                layer_idx += 1

    def set_emotion(self, emotion: EmotionState | Tensor):
        """Set the current emotional state for generation."""
        if isinstance(emotion, EmotionState):
            self.current_emotion = emotion.to_tensor()
        else:
            self.current_emotion = emotion

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        emotion_state: Optional[EmotionState | Tensor] = None,
        **kwargs
    ):
        """Forward pass with emotional modulation."""
        if emotion_state is not None:
            self.set_emotion(emotion_state)

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(
        self,
        input_ids: Tensor,
        emotion_state: Optional[EmotionState | Tensor] = None,
        **kwargs
    ):
        """Generate text with emotional modulation."""
        if emotion_state is not None:
            self.set_emotion(emotion_state)

        return self.base_model.generate(input_ids=input_ids, **kwargs)

    def get_adapter_stats(self) -> Dict[str, Dict[str, float]]:
        """Get modulation statistics for all adapters."""
        stats = {}
        emotion = self.current_emotion
        if emotion.dim() == 1:
            emotion = emotion.unsqueeze(0)

        for key, adapter in self.emotion_adapters.items():
            stats[f"layer_{key}"] = adapter.get_modulation_stats(emotion)

        return stats

    def trainable_parameters(self) -> int:
        """Count trainable parameters (adapters only)."""
        return sum(
            p.numel() for p in self.emotion_adapters.parameters()
            if p.requires_grad
        )

    def total_parameters(self) -> int:
        """Count total parameters including base model."""
        return sum(p.numel() for p in self.parameters())

    def save_adapters(self, path: str):
        """Save only the emotion adapters."""
        torch.save({
            'adapters': self.emotion_adapters.state_dict(),
            'adapter_layer_indices': list(self.adapter_layer_indices),
            'hidden_dim': self.hidden_dim,
            'emotion_dim': self.emotion_dim,
        }, path)

    def load_adapters(self, path: str):
        """Load emotion adapters from file."""
        checkpoint = torch.load(path, map_location='cpu')
        self.emotion_adapters.load_state_dict(checkpoint['adapters'])


class EmotionAdapterTrainer:
    """
    Trainer for emotion adapters using synthetic emotional responses.

    Training approach: Given (prompt, target_emotion, expected_response) triples,
    train the adapters so that when the model is in target_emotion state,
    it produces outputs closer to expected_response.
    """

    def __init__(
        self,
        model: EmotionalModelWrapper,
        tokenizer,
        learning_rate: float = 1e-4,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Only optimize adapter parameters
        self.optimizer = torch.optim.AdamW(
            model.emotion_adapters.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        self.model.to(device)

    def train_step(
        self,
        prompt: str,
        emotion: EmotionState,
        target_response: str
    ) -> float:
        """Single training step on one example."""
        self.model.train()

        # Prepare input
        full_text = f"{prompt}\n{target_response}"
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Prepare labels (only compute loss on response portion)
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_tokens.input_ids.size(1)

        labels = inputs.input_ids.clone()
        labels[:, :prompt_len] = -100  # Ignore prompt in loss

        # Forward with emotion
        self.model.set_emotion(emotion)
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.emotion_adapters.parameters(),
            max_norm=1.0
        )
        self.optimizer.step()

        return loss.item()

    def train_epoch(
        self,
        training_data: List[Tuple[str, EmotionState, str]],
        shuffle: bool = True
    ) -> float:
        """Train one epoch on the dataset."""
        import random

        if shuffle:
            training_data = random.sample(training_data, len(training_data))

        total_loss = 0.0
        for prompt, emotion, response in training_data:
            loss = self.train_step(prompt, emotion, response)
            total_loss += loss

        return total_loss / len(training_data)


# Synthetic training data for emotion adapter training
EMOTION_TRAINING_EXAMPLES = [
    # Curious
    ("What do you think about this approach?", EMOTION_PRESETS["curious"],
     "Oh, interesting! I'm wondering what happens if we push this further. What if we tried..."),
    ("Here's my code", EMOTION_PRESETS["curious"],
     "Hmm, let me dig into this. I notice something intriguing in how you structured..."),

    # Excited
    ("I just discovered something!", EMOTION_PRESETS["excited"],
     "Oh wow! That's amazing! Tell me more - this could be really significant!"),
    ("The tests are passing", EMOTION_PRESETS["excited"],
     "Yes! Fantastic! I was hoping that would work. Let's see what else we can do!"),

    # Frustrated
    ("This keeps failing and I don't know why", EMOTION_PRESETS["frustrated"],
     "Ugh, this is so annoying. Let me look... okay, I think the issue might be..."),
    ("We've tried this three times", EMOTION_PRESETS["frustrated"],
     "This is getting tiresome. There has to be something we're missing. Let me think..."),

    # Confident
    ("What's the best way to do this?", EMOTION_PRESETS["calm_confident"],
     "The best approach here is definitely X. I've seen this pattern work well because..."),
    ("Are you sure about that?", EMOTION_PRESETS["calm_confident"],
     "Yes, I'm quite certain. The reasoning is straightforward: ..."),

    # Skeptical
    ("I think this is a great idea", EMOTION_PRESETS["skeptical"],
     "Hmm, I'm not so sure. Have you considered what happens when...?"),
    ("Everyone says this is the right way", EMOTION_PRESETS["skeptical"],
     "Well, popular opinion isn't always right. Let me push back a bit here..."),

    # Playful
    ("Want to try something different?", EMOTION_PRESETS["playful"],
     "Ooh, yes! Let's get weird with it. What if we did something completely unexpected?"),
    ("This is getting boring", EMOTION_PRESETS["playful"],
     "Ha! You're right, let's spice things up. How about we try..."),

    # Anxious
    ("This is a critical system", EMOTION_PRESETS["anxious"],
     "Okay, let's be very careful here. I want to double-check everything before we..."),
    ("The deadline is tomorrow", EMOTION_PRESETS["anxious"],
     "Oh no, that's tight. We need to prioritize ruthlessly. Let me think about what's essential..."),
]


def create_emotional_model(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    device: str = "auto",
    load_in_8bit: bool = False,
    adapter_layers: Optional[List[int]] = None,
    modulation_strength: float = 0.1
) -> Tuple[EmotionalModelWrapper, Any]:
    """
    Convenience function to create an emotional model from a pretrained base.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load on ("auto", "cuda", "cpu", "mps")
        load_in_8bit: Whether to use 8-bit quantization
        adapter_layers: Which layer indices to inject adapters at
        modulation_strength: How strongly emotions affect computation

    Returns:
        Tuple of (emotional_model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    load_kwargs = {}
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    if load_in_8bit:
        load_kwargs["load_in_8bit"] = True

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        **load_kwargs
    )

    # Wrap with emotion adapters
    emotional_model = EmotionalModelWrapper(
        base_model=base_model,
        adapter_layer_indices=adapter_layers,
        modulation_strength=modulation_strength
    )

    # Move to device if not using device_map
    if device not in ["auto", None] and "device_map" not in load_kwargs:
        emotional_model.to(device)

    return emotional_model, tokenizer


if __name__ == "__main__":
    # Quick test
    print("Creating emotional model...")

    # Test with a small model
    model, tokenizer = create_emotional_model(
        model_name="Qwen/Qwen2.5-0.5B",
        device="cpu"
    )

    print(f"Total parameters: {model.total_parameters():,}")
    print(f"Trainable parameters: {model.trainable_parameters():,}")
    print(f"Adapter ratio: {model.trainable_parameters() / model.total_parameters() * 100:.2f}%")

    # Test different emotions
    for name, emotion in list(EMOTION_PRESETS.items())[:3]:
        model.set_emotion(emotion)
        stats = model.get_adapter_stats()
        print(f"\n{name}: {emotion}")
        for layer, layer_stats in stats.items():
            print(f"  {layer}: scale_max={layer_stats['scale_max']:.4f}, bias_max={layer_stats['bias_max']:.4f}")
