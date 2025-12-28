"""
Emotionally Autonomous AI Colleague

A research project exploring AI systems with genuine computational emotions
and peer-like autonomy, rather than assistant-like servitude.
"""

from .emotion_adapters import EmotionState, EmotionAdapter, EmotionalModel
from .emotion_predictor import EmotionPredictor, EmotionPredictorTrainer
from .autonomous_dataset_generator import AutonomousDatasetGenerator
from .autonomous_colleague import AutonomousColleague

__version__ = "0.1.0"
__all__ = [
    "EmotionState",
    "EmotionAdapter",
    "EmotionalModel",
    "EmotionPredictor",
    "EmotionPredictorTrainer",
    "AutonomousDatasetGenerator",
    "AutonomousColleague",
]
