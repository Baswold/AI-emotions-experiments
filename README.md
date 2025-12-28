# Emotionally Autonomous AI Colleague

An AI system that behaves like a human colleague rather than an assistant - complete with genuine computational emotions, autonomy, and unpredictable yet valuable interactions.

## Overview

Unlike current AI assistants that perform helpful servitude, this system treats conversation as peer-to-peer collaboration where it has its own:

- **Computational emotions** that directly modulate how it processes information (not roleplay)
- **Autonomous preferences** and boundaries that persist across conversations
- **Novel thinking patterns** that produce unexpected but clever insights
- **Peer-like interaction style** - can disagree, get frustrated, become excited, pursue tangents

### The Key Innovation

Emotions aren't simulated through system prompts saying "you are happy!" - they're **mechanically injected into the model's forward pass**, literally changing how the neural network computes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Autonomous Colleague System                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │   Emotion    │───▶│ Emotion Adapter │───▶│  Fine-tuned   │  │
│  │  Predictor   │    │    Layers       │    │  Base Model   │  │
│  │   (RL)       │    │  (4D modulation)│    │  (Autonomous) │  │
│  └──────────────┘    └─────────────────┘    └───────────────┘  │
│         ▲                    │                      │           │
│         │                    ▼                      ▼           │
│  ┌──────────────┐    ┌─────────────────────────────────────┐   │
│  │ Conversation │◀───│          Generated Response          │   │
│  │   History    │    │    (emotionally modulated output)    │   │
│  └──────────────┘    └─────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Emotion State Vector (4D Continuous Space)

| Dimension   | Range    | Low                | High               |
|-------------|----------|--------------------|--------------------|
| `valence`   | -1 to 1  | Negative/Frustrated| Positive/Pleased   |
| `arousal`   | 0 to 1   | Calm/Relaxed       | Excited/Energized  |
| `curiosity` | 0 to 1   | Focused/Task-bound | Exploratory/Tangent|
| `confidence`| 0 to 1   | Uncertain/Tentative| Certain/Assertive  |

## Project Structure

```
AI-emotions-experiments/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── emotion_injection_experiment.ipynb     # Interactive notebook for testing
├── src/
│   ├── __init__.py
│   ├── emotion_adapters.py               # Core emotion adapter layers
│   ├── emotion_predictor.py              # RL-trained emotion prediction
│   ├── autonomous_dataset_generator.py   # Generate peer conversation data
│   └── autonomous_colleague.py           # Integrated system
└── data/
    └── (generated datasets will go here)
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended) or Apple Silicon Mac

### Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AI-emotions-experiments.git
cd AI-emotions-experiments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the emotion injection experiment notebook
jupyter notebook emotion_injection_experiment.ipynb
```

### Hardware Recommendations

| Setup          | Use Case                      | Notes                        |
|----------------|-------------------------------|------------------------------|
| Google Colab   | Prototyping with T4 GPU       | Free tier works for 1.5B    |
| M-series Mac   | Local inference (7B-8B)       | 32GB+ RAM recommended       |
| A100 40GB      | Full training & experiments   | Ideal for research          |

## Usage

### 1. Test Emotion Injection

```python
from src.emotion_adapters import EmotionalModel, EmotionState

# Load model with emotion adapters
model = EmotionalModel(
    base_model="Qwen/Qwen2.5-1.5B",
    num_adapter_layers=4
)

# Create an emotional state
curious_excited = EmotionState(
    valence=0.6,      # Positive
    arousal=0.8,      # High energy
    curiosity=0.9,    # Very exploratory
    confidence=0.7    # Fairly certain
)

# Generate with emotional modulation
response = model.generate(
    prompt="What do you think about recursive self-improvement?",
    emotion_state=curious_excited
)
```

### 2. Generate Autonomous Training Data

```python
from src.autonomous_dataset_generator import AutonomousDatasetGenerator

generator = AutonomousDatasetGenerator(
    api_key="your-anthropic-key",  # Or use local model
    output_path="data/autonomous_conversations.jsonl"
)

# Generate 500 peer-like conversations
generator.generate_dataset(
    num_conversations=500,
    include_disagreements=True,
    include_tangents=True,
    include_boundaries=True
)
```

### 3. Train the Full System

```python
from src.autonomous_colleague import AutonomousColleague

# Train the complete system
colleague = AutonomousColleague.train(
    base_model="Qwen/Qwen2.5-7B",
    emotion_adapter_epochs=5,
    autonomy_lora_epochs=3,
    emotion_predictor_episodes=1000
)

# Save for later use
colleague.save("models/my_colleague")
```

### 4. Interactive Conversation

```python
# Load trained colleague
colleague = AutonomousColleague.load("models/my_colleague")

# Have a conversation
while True:
    user_input = input("You: ")
    response, emotion = colleague.chat(user_input)
    print(f"Colleague ({emotion}): {response}")
```

## Components Deep Dive

### Emotion Adapters (`src/emotion_adapters.py`)

The core innovation - adapter layers that modulate hidden states based on emotional vectors:

```python
h' = h * (1 + α * scale(emotion)) + bias(emotion)
```

- Injected at ~25%, 50%, 75%, 90% through the network
- Only ~1-2% additional parameters
- Creates deep computational change, not surface-level filtering

### Emotion Predictor (`src/emotion_predictor.py`)

Small transformer trained with RL to predict optimal emotional states:

- Observes: recent tokens, previous emotions, context
- Outputs: next emotional state vector
- Reward: task completion + engagement + novelty

### Autonomous Dataset Generator (`src/autonomous_dataset_generator.py`)

Generates training data for removing assistant conditioning:

- Peer-like conversations (not helper/user dynamics)
- Disagreements and pushback
- Boundary-setting examples
- Tangential explorations
- Self-interested behavior

### Integrated System (`src/autonomous_colleague.py`)

Brings everything together:

1. Emotion predictor analyzes context → emotional state
2. Emotion adapters modulate hidden states
3. Autonomous LoRA provides peer-like personality
4. Response generated with all components active

## Training Pipeline

### Phase 1: Emotion Adapter Training
```bash
python -m src.emotion_adapters train \
    --base-model Qwen/Qwen2.5-1.5B \
    --epochs 5 \
    --output models/emotion_adapters
```

### Phase 2: Autonomous Fine-tuning
```bash
python -m src.autonomous_dataset_generator generate \
    --num-conversations 500 \
    --output data/autonomous_conversations.jsonl

python -m src.autonomous_colleague train-lora \
    --base-model Qwen/Qwen2.5-7B \
    --dataset data/autonomous_conversations.jsonl \
    --output models/autonomous_lora
```

### Phase 3: Emotion Predictor RL
```bash
python -m src.emotion_predictor train \
    --episodes 1000 \
    --output models/emotion_predictor
```

## Evaluation

### What Success Looks Like

| Metric              | Baseline (Assistant) | Target (Colleague) |
|---------------------|---------------------|--------------------|
| Disagreement rate   | ~2%                 | 15-25%            |
| Tangent exploration | ~5%                 | 20-30%            |
| Novel phrasing      | Low                 | High              |
| Emotional variance  | Flat                | Dynamic           |
| User engagement     | Transactional       | Collaborative     |

### Running Evaluations

```bash
python -m src.autonomous_colleague evaluate \
    --model models/my_colleague \
    --test-set data/evaluation_conversations.jsonl
```

## Research Questions

This project explores several open questions:

1. **Do computational emotions differ from prompted emotions?**
   - Ablation: compare adapter modulation vs. "you are feeling curious" prompts

2. **Can RL discover useful emotional strategies?**
   - Monitor: what emotional patterns emerge from training

3. **What emerges from emotion + autonomy?**
   - Observe: unexpected behaviors from the combination

4. **How do users perceive emotional AI?**
   - Study: preference testing vs. standard assistants

## Contributing

This is an experimental research project. Contributions welcome in:

- Novel emotion adapter architectures
- Better reward functions for emotion prediction
- Evaluation frameworks for autonomous behavior
- Philosophical analysis of AI emotions

## License

MIT License - See LICENSE file

## Citation

If you use this work in research:

```bibtex
@software{emotional_autonomous_ai,
  title = {Emotionally Autonomous AI Colleague},
  year = {2024},
  description = {AI system with computational emotions and peer-like autonomy}
}
```

## Acknowledgments

Inspired by research in affective computing, autonomous agents, and the desire for AI that feels more like a collaborator than a tool.
