"""Agent Evolution System — Self-improvement through execution feedback."""

from huginn.evolution.engine import EvolutionEngine
from huginn.evolution.logger import ExecutionLogger
from huginn.evolution.skill_evolver import SkillEvolver
from huginn.evolution.knowledge_distiller import KnowledgeDistiller

__all__ = ["EvolutionEngine", "ExecutionLogger", "SkillEvolver", "KnowledgeDistiller"]
