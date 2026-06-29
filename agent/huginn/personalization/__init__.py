"""个人定制模块 — 学习用户语言偏好, 逐步定制 agent 通信风格.

StyleLearner 实例通过共享单例在 agent / route / tool 之间流转,
保证同一用户的 profile 全局一致. profile 只存本地 SQLite, 不上传.
"""

from __future__ import annotations

from pathlib import Path

from huginn.personalization.user_style import StyleLearner, UserStyleProfile

__all__ = [
    "StyleLearner",
    "UserStyleProfile",
    "get_shared_style_learner",
    "set_shared_style_learner",
]

_shared_learner: StyleLearner | None = None


def get_shared_style_learner() -> StyleLearner:
    """拿共享的 StyleLearner 单例.

    没有就懒创建一个, 默认存到 workspace 下 style_profile.db.
    agent / route / tool 都走这个入口, 保证 profile 全局一致.
    """
    global _shared_learner
    if _shared_learner is None:
        path = _default_storage_path()
        _shared_learner = StyleLearner(path)
    return _shared_learner


def set_shared_style_learner(learner: StyleLearner | None) -> None:
    """注入外部 StyleLearner, 覆盖懒创建的实例.

    agent.set_style_learner() 会调这个, 把自己的 learner 注册成全局共享.
    """
    global _shared_learner
    _shared_learner = learner


def _default_storage_path() -> str:
    """默认把 profile 存到 workspace 下, 跟其他状态文件放一起."""
    try:
        from huginn.config import HuginnConfig

        cfg = HuginnConfig.from_env()
        if cfg.workspace:
            p = Path(cfg.workspace) / "style_profile.db"
            p.parent.mkdir(parents=True, exist_ok=True)
            return str(p)
    except Exception:
        pass
    return ":memory:"
