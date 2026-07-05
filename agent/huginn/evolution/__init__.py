"""Agent Evolution System — Self-improvement through execution feedback."""

from huginn.evolution.engine import EvolutionEngine
from huginn.evolution.knowledge_distiller import DistilledKnowledge, KnowledgeDistiller
from huginn.evolution.logger import ExecutionLogger

__all__ = [
    "EvolutionEngine",
    "ExecutionLogger",
    "DistilledKnowledge",
    "KnowledgeDistiller",
]
