"""
RL-Trained Emotion Predictor

This module implements a small transformer that learns to predict optimal
emotional states through reinforcement learning. The predictor observes
conversation context and previous emotions, then outputs the next emotional
state that maximizes long-term conversational reward.

Key insight: Rather than mimicking human emotions, this learns *useful*
emotional strategies - potentially discovering non-human patterns that
work better for AI cognition.

Training approach: PPO-style RL where:
- States: conversation context + previous emotions
- Actions: next emotional state (continuous 4D vector)
- Rewards: task completion + engagement + novelty + appropriateness
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal


@dataclass
class EmotionPredictorConfig:
    """Configuration for the emotion predictor."""
    context_dim: int = 768          # Dimension of context embeddings
    emotion_dim: int = 4            # Dimension of emotion vector
    hidden_dim: int = 256           # Transformer hidden dimension
    num_heads: int = 4              # Attention heads
    num_layers: int = 2             # Transformer layers
    max_context_length: int = 32    # Max number of context tokens to consider
    dropout: float = 0.1

    # RL hyperparameters
    gamma: float = 0.99             # Discount factor
    gae_lambda: float = 0.95        # GAE lambda
    clip_epsilon: float = 0.2       # PPO clip range
    entropy_coef: float = 0.01      # Entropy bonus coefficient
    value_coef: float = 0.5         # Value loss coefficient
    max_grad_norm: float = 0.5      # Gradient clipping


class ContextEncoder(nn.Module):
    """
    Encodes conversation context into a fixed-size representation.

    Uses a small transformer to attend over recent conversation history
    and produce a context embedding that captures the current situation.
    """

    def __init__(self, config: EmotionPredictorConfig):
        super().__init__()
        self.config = config

        # Project input embeddings to hidden dim
        self.input_proj = nn.Linear(config.context_dim, config.hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_context_length, config.hidden_dim) * 0.02
        )

        # Output projection (mean pooling + projection)
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(
        self,
        context_embeddings: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Encode context into fixed-size representation.

        Args:
            context_embeddings: (batch, seq_len, context_dim)
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding

        Returns:
            Context representation of shape (batch, hidden_dim)
        """
        batch_size, seq_len, _ = context_embeddings.shape

        # Project and add positional encoding
        x = self.input_proj(context_embeddings)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Create attention mask for transformer (True = masked out)
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        # Encode
        encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling (masked)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed = (encoded * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        else:
            pooled = encoded.mean(dim=1)

        return self.output_proj(pooled)


class EmotionPredictor(nn.Module):
    """
    Predicts the next emotional state given context and previous emotion.

    This is the "actor" in our actor-critic setup. It outputs a distribution
    over the 4D emotion space (continuous action space).

    Architecture:
        context_encoder(conversation) + emotion_history -> MLP -> μ, σ for each emotion dim
    """

    def __init__(self, config: EmotionPredictorConfig):
        super().__init__()
        self.config = config

        # Context encoder
        self.context_encoder = ContextEncoder(config)

        # Emotion history encoder (simple MLP for now)
        self.emotion_history_proj = nn.Sequential(
            nn.Linear(config.emotion_dim * 3, config.hidden_dim),  # Last 3 emotions
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # Combine context and emotion history
        self.combine = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # Output heads for mean and log_std of each emotion dimension
        self.mean_head = nn.Linear(config.hidden_dim, config.emotion_dim)
        self.log_std_head = nn.Linear(config.hidden_dim, config.emotion_dim)

        # Initialize outputs for reasonable starting emotions
        nn.init.zeros_(self.mean_head.bias)
        nn.init.constant_(self.log_std_head.bias, -1.0)  # Start with small variance

    def forward(
        self,
        context_embeddings: Tensor,
        emotion_history: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict next emotion distribution.

        Args:
            context_embeddings: (batch, seq_len, context_dim) - conversation context
            emotion_history: (batch, 3, emotion_dim) - last 3 emotional states
            attention_mask: (batch, seq_len)

        Returns:
            mean: (batch, emotion_dim) - mean of emotion distribution
            log_std: (batch, emotion_dim) - log std of emotion distribution
        """
        # Encode context
        context_repr = self.context_encoder(context_embeddings, attention_mask)

        # Encode emotion history
        emotion_flat = emotion_history.view(emotion_history.size(0), -1)
        emotion_repr = self.emotion_history_proj(emotion_flat)

        # Combine
        combined = torch.cat([context_repr, emotion_repr], dim=-1)
        features = self.combine(combined)

        # Output distribution parameters
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)

        return mean, log_std

    def sample(
        self,
        context_embeddings: Tensor,
        emotion_history: Tensor,
        attention_mask: Optional[Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample an emotion from the predicted distribution.

        Args:
            deterministic: If True, return mean instead of sampling

        Returns:
            emotion: (batch, emotion_dim) - sampled emotion
            log_prob: (batch,) - log probability of the sample
        """
        mean, log_std = self(context_embeddings, emotion_history, attention_mask)
        std = log_std.exp()

        if deterministic:
            emotion = mean
            log_prob = torch.zeros(mean.size(0), device=mean.device)
        else:
            # Sample from Gaussian
            dist = Normal(mean, std)
            emotion = dist.rsample()  # Reparameterized sample
            log_prob = dist.log_prob(emotion).sum(dim=-1)

        # Clamp to valid ranges
        # valence: tanh for [-1, 1]
        # arousal, curiosity, confidence: sigmoid for [0, 1]
        clamped = torch.zeros_like(emotion)
        clamped[:, 0] = torch.tanh(emotion[:, 0])       # valence
        clamped[:, 1:] = torch.sigmoid(emotion[:, 1:])  # others

        return clamped, log_prob

    def evaluate_action(
        self,
        context_embeddings: Tensor,
        emotion_history: Tensor,
        action: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Evaluate log probability and entropy of a given action.

        Used during PPO training to compute policy gradient.
        """
        mean, log_std = self(context_embeddings, emotion_history, attention_mask)
        std = log_std.exp()

        dist = Normal(mean, std)

        # Transform action back to unbounded space for log prob
        unbounded = torch.zeros_like(action)
        unbounded[:, 0] = torch.atanh(action[:, 0].clamp(-0.999, 0.999))
        unbounded[:, 1:] = torch.log(
            (action[:, 1:].clamp(0.001, 0.999)) /
            (1 - action[:, 1:].clamp(0.001, 0.999))
        )

        log_prob = dist.log_prob(unbounded).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class ValueNetwork(nn.Module):
    """
    Critic network that estimates expected return from a state.

    Used in PPO to compute advantages and reduce variance in policy gradient.
    """

    def __init__(self, config: EmotionPredictorConfig):
        super().__init__()
        self.config = config

        # Share context encoder architecture (but separate weights)
        self.context_encoder = ContextEncoder(config)

        self.emotion_history_proj = nn.Sequential(
            nn.Linear(config.emotion_dim * 3, config.hidden_dim),
            nn.ReLU()
        )

        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(
        self,
        context_embeddings: Tensor,
        emotion_history: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Estimate value of current state.

        Returns:
            value: (batch, 1)
        """
        context_repr = self.context_encoder(context_embeddings, attention_mask)
        emotion_flat = emotion_history.view(emotion_history.size(0), -1)
        emotion_repr = self.emotion_history_proj(emotion_flat)

        combined = torch.cat([context_repr, emotion_repr], dim=-1)
        return self.value_head(combined)


@dataclass
class Episode:
    """A single training episode (conversation)."""
    context_embeddings: List[Tensor]  # Context at each step
    emotion_history: List[Tensor]     # Emotion history at each step
    actions: List[Tensor]             # Emotions chosen
    rewards: List[float]              # Rewards received
    log_probs: List[Tensor]           # Log probs of actions
    values: List[Tensor]              # Value estimates


class RewardModel:
    """
    Computes rewards for emotional states in context.

    Reward components:
    1. Task performance: Did the response help solve the problem?
    2. Emotional appropriateness: Does the emotion match the context?
    3. Conversational engagement: Is the interaction engaging?
    4. Novelty: Is the response interesting/unexpected?
    """

    def __init__(
        self,
        task_weight: float = 0.4,
        appropriateness_weight: float = 0.2,
        engagement_weight: float = 0.2,
        novelty_weight: float = 0.2
    ):
        self.weights = {
            'task': task_weight,
            'appropriateness': appropriateness_weight,
            'engagement': engagement_weight,
            'novelty': novelty_weight
        }

        # Simple heuristics for reward computation
        # In practice, these could be learned reward models
        self.context_emotion_mapping = {
            'question': {'curiosity': 0.7, 'confidence': 0.5},
            'error': {'valence': -0.3, 'arousal': 0.6},
            'success': {'valence': 0.7, 'arousal': 0.7},
            'confusion': {'curiosity': 0.8, 'confidence': 0.3},
            'neutral': {'valence': 0.0, 'arousal': 0.5},
        }

    def compute_reward(
        self,
        context_type: str,
        emotion: Tensor,
        response_quality: float,
        novelty_score: float
    ) -> float:
        """
        Compute total reward for an emotional response.

        Args:
            context_type: Type of context ('question', 'error', etc.)
            emotion: The emotional state that was used
            response_quality: How good the generated response was (0-1)
            novelty_score: How novel/interesting the response was (0-1)

        Returns:
            Total reward
        """
        rewards = {}

        # Task performance
        rewards['task'] = response_quality

        # Emotional appropriateness
        target = self.context_emotion_mapping.get(context_type, {})
        emotion_np = emotion.detach().cpu().numpy()

        appropriateness = 1.0
        dim_names = ['valence', 'arousal', 'curiosity', 'confidence']
        for i, dim in enumerate(dim_names):
            if dim in target:
                diff = abs(emotion_np[i] - target[dim])
                appropriateness -= diff * 0.25
        rewards['appropriateness'] = max(0, appropriateness)

        # Engagement (placeholder - would use actual engagement metrics)
        rewards['engagement'] = 0.5 + 0.3 * emotion_np[1]  # Higher arousal = more engaging

        # Novelty
        rewards['novelty'] = novelty_score

        # Weighted sum
        total = sum(
            self.weights[key] * rewards[key]
            for key in self.weights
        )

        return total


class EmotionPredictorTrainer:
    """
    PPO trainer for the emotion predictor.

    Training loop:
    1. Collect episodes using current policy
    2. Compute advantages using GAE
    3. Update policy and value networks using PPO objective
    """

    def __init__(
        self,
        config: EmotionPredictorConfig,
        policy: EmotionPredictor,
        value_net: ValueNetwork,
        reward_model: RewardModel,
        lr: float = 3e-4,
        device: str = "cuda"
    ):
        self.config = config
        self.policy = policy.to(device)
        self.value_net = value_net.to(device)
        self.reward_model = reward_model
        self.device = device

        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)

    def compute_gae(
        self,
        rewards: List[float],
        values: List[Tensor],
        next_value: float = 0.0
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation.

        Returns advantages and returns for policy gradient.
        """
        gamma = self.config.gamma
        lam = self.config.gae_lambda

        advantages = []
        returns = []

        gae = 0
        next_val = next_value

        # Work backwards
        for t in reversed(range(len(rewards))):
            current_val = values[t].item()
            delta = rewards[t] + gamma * next_val - current_val
            gae = delta + gamma * lam * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + current_val)

            next_val = current_val

        return advantages, returns

    def collect_episode(
        self,
        conversation_contexts: List[Tuple[Tensor, str]],  # (context_embedding, context_type)
        response_generator  # Callable that generates response and quality score
    ) -> Episode:
        """
        Collect a single episode by running policy on a conversation.

        Args:
            conversation_contexts: List of (context_embedding, context_type) tuples
            response_generator: Function that takes emotion and context, returns
                               (response, quality_score, novelty_score)
        """
        self.policy.eval()
        self.value_net.eval()

        episode = Episode(
            context_embeddings=[],
            emotion_history=[],
            actions=[],
            rewards=[],
            log_probs=[],
            values=[]
        )

        # Initialize emotion history (last 3 emotions, starting neutral)
        emotion_history = torch.zeros(3, self.config.emotion_dim, device=self.device)

        with torch.no_grad():
            for context_emb, context_type in conversation_contexts:
                context_emb = context_emb.to(self.device)

                # Prepare inputs
                context_batch = context_emb.unsqueeze(0)  # Add batch dim
                history_batch = emotion_history.unsqueeze(0)

                # Sample emotion from policy
                emotion, log_prob = self.policy.sample(
                    context_batch,
                    history_batch
                )

                # Get value estimate
                value = self.value_net(context_batch, history_batch)

                # Generate response with this emotion
                response, quality, novelty = response_generator(
                    emotion.squeeze(0),
                    context_emb
                )

                # Compute reward
                reward = self.reward_model.compute_reward(
                    context_type,
                    emotion.squeeze(0),
                    quality,
                    novelty
                )

                # Store in episode
                episode.context_embeddings.append(context_emb)
                episode.emotion_history.append(emotion_history.clone())
                episode.actions.append(emotion.squeeze(0))
                episode.rewards.append(reward)
                episode.log_probs.append(log_prob.squeeze(0))
                episode.values.append(value.squeeze())

                # Update emotion history (FIFO)
                emotion_history = torch.cat([
                    emotion_history[1:],
                    emotion.squeeze(0).unsqueeze(0)
                ], dim=0)

        return episode

    def train_step(self, episodes: List[Episode], ppo_epochs: int = 4) -> Dict[str, float]:
        """
        Perform PPO update on collected episodes.
        """
        self.policy.train()
        self.value_net.train()

        # Flatten episodes
        all_contexts = []
        all_histories = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []

        for episode in episodes:
            advantages, returns = self.compute_gae(episode.rewards, episode.values)

            all_contexts.extend(episode.context_embeddings)
            all_histories.extend(episode.emotion_history)
            all_actions.extend(episode.actions)
            all_old_log_probs.extend(episode.log_probs)
            all_advantages.extend(advantages)
            all_returns.extend(returns)

        # Convert to tensors
        contexts = torch.stack(all_contexts)
        histories = torch.stack(all_histories)
        actions = torch.stack(all_actions)
        old_log_probs = torch.stack(all_old_log_probs)
        advantages = torch.tensor(all_advantages, device=self.device)
        returns = torch.tensor(all_returns, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(ppo_epochs):
            # Evaluate current policy on old actions
            new_log_probs, entropy = self.policy.evaluate_action(
                contexts.unsqueeze(1),  # Add seq dim
                histories,
                actions
            )

            # Policy loss (clipped surrogate objective)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = self.value_net(contexts.unsqueeze(1), histories).squeeze()
            value_loss = F.mse_loss(values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                policy_loss +
                self.config.value_coef * value_loss +
                self.config.entropy_coef * entropy_loss
            )

            # Update
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.value_net.parameters(),
                self.config.max_grad_norm
            )

            self.policy_optimizer.step()
            self.value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        return {
            'policy_loss': total_policy_loss / ppo_epochs,
            'value_loss': total_value_loss / ppo_epochs,
            'entropy': total_entropy / ppo_epochs,
            'mean_reward': sum(
                sum(ep.rewards) / len(ep.rewards)
                for ep in episodes
            ) / len(episodes)
        }

    def save(self, path: str):
        """Save policy and value network."""
        torch.save({
            'policy': self.policy.state_dict(),
            'value_net': self.value_net.state_dict(),
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load policy and value network."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value_net.load_state_dict(checkpoint['value_net'])


def create_emotion_predictor(
    context_dim: int = 768,
    device: str = "cuda"
) -> Tuple[EmotionPredictor, ValueNetwork, EmotionPredictorConfig]:
    """
    Create emotion predictor and value network.

    Returns:
        Tuple of (policy, value_net, config)
    """
    config = EmotionPredictorConfig(context_dim=context_dim)

    policy = EmotionPredictor(config).to(device)
    value_net = ValueNetwork(config).to(device)

    return policy, value_net, config


if __name__ == "__main__":
    # Quick test
    print("Testing emotion predictor...")

    config = EmotionPredictorConfig(context_dim=768)
    policy = EmotionPredictor(config)
    value_net = ValueNetwork(config)

    # Create dummy inputs
    batch_size = 4
    seq_len = 16
    context = torch.randn(batch_size, seq_len, config.context_dim)
    emotion_history = torch.randn(batch_size, 3, config.emotion_dim)

    # Test forward pass
    emotion, log_prob = policy.sample(context, emotion_history)
    value = value_net(context, emotion_history)

    print(f"Sampled emotion shape: {emotion.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Sample emotion: {emotion[0].tolist()}")
    print(f"  Valence: {emotion[0, 0]:.3f} (should be in [-1, 1])")
    print(f"  Arousal: {emotion[0, 1]:.3f} (should be in [0, 1])")
    print(f"  Curiosity: {emotion[0, 2]:.3f} (should be in [0, 1])")
    print(f"  Confidence: {emotion[0, 3]:.3f} (should be in [0, 1])")
