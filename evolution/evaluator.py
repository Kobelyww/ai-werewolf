"""Hermes 风格适应度评估 + 约束门控（对标 s26）。

agent 表现 = 模型能力 × 上下文文本质量

s23（RL训练）改进模型，本模块改进文本——系统性地评估和约束
System Prompt、策略指令等上下文文本的质量，不改模型权重。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from langchain_core.language_models import BaseChatModel


class FitnessDimension(str, Enum):
    DECEPTION = "伪装能力"         # 狼人：不被识破的能力
    LOGIC = "逻辑自洽"             # 发言逻辑一致性
    PERSUASION = "说服力"          # 能否说服他人
    COORDINATION = "协作效率"      # 狼人内部协调 / 好人阵营配合
    INFORMATION_USE = "信息利用"   # 对已知信息的利用程度
    ADAPTABILITY = "适应性"        # 对局势变化的应对
    ROLE_SPECIFIC = "角色专精"     # 角色特定能力(预言家查验、女巫用药等)


@dataclass
class FitnessScore:
    """多维适应度分数。"""
    role: str
    overall: float                           # 0-1 综合分
    dimensions: dict[FitnessDimension, float]  # 各维度得分
    raw_win_rate: float                      # 原始胜率
    sample_size: int                         # 评估局数
    details: str = ""                        # LLM 评分理由

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "overall": self.overall,
            "dimensions": {d.value: s for d, s in self.dimensions.items()},
            "win_rate": self.raw_win_rate,
            "samples": self.sample_size,
        }


@dataclass
class ConstraintCheck:
    """约束检查结果。"""
    passed: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    # (check_name, passed, detail)

    @property
    def summary(self) -> str:
        lines = []
        for name, ok, detail in self.checks:
            lines.append(f"  {'✓' if ok else '✗'} {name}: {detail}")
        return "\n".join(lines)


FITNESS_PROMPT = """你是狼人杀 AI 行为评审专家。请根据以下角色在游戏中的表现，按 7 个维度打分（每项 1-5 分，5 最优）。

角色：{role}
胜率：{win_rate:.0%}
对局数：{sample_size}

## 游戏事件摘要
{game_summary}

## 评分维度
1. 伪装能力 (1-5)：不让对手识破身份的能力（狼人越高越好，好人不需要）
2. 逻辑自洽 (1-5)：发言逻辑是否一致、无自相矛盾
3. 说服力 (1-5)：言论是否影响了投票结果
4. 协作效率 (1-5)：与同伴的配合程度
5. 信息利用 (1-5)：对已知信息的利用效率
6. 适应性 (1-5)：对局势变化的反应能力
7. 角色专精 (1-5)：特定角色技能的发挥（预言家查验、女巫用药等）

请输出：
整体评分（1-5 分，含一位小数）：
伪装能力：
逻辑自洽：
说服力：
协作效率：
信息利用：
适应性：
角色专精：
评分理由（2-3 句话）："""


class FitnessEvaluator:
    """LLM-as-judge 多维适应度评估器。"""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def evaluate(
        self,
        role: str,
        game_logs: list[dict],
        win_rate: float,
    ) -> FitnessScore:
        """对角色进行多维适应度评估。"""
        summary = self._summarize_games(game_logs, role)

        prompt = FITNESS_PROMPT.format(
            role=role, win_rate=win_rate, sample_size=len(game_logs), game_summary=summary
        )
        result = self.llm.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)
        if isinstance(content, list):
            content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)

        dims = {
            FitnessDimension.DECEPTION: self._extract_score(content, "伪装能力"),
            FitnessDimension.LOGIC: self._extract_score(content, "逻辑自洽"),
            FitnessDimension.PERSUASION: self._extract_score(content, "说服力"),
            FitnessDimension.COORDINATION: self._extract_score(content, "协作效率"),
            FitnessDimension.INFORMATION_USE: self._extract_score(content, "信息利用"),
            FitnessDimension.ADAPTABILITY: self._extract_score(content, "适应性"),
            FitnessDimension.ROLE_SPECIFIC: self._extract_score(content, "角色专精"),
        }

        overall = self._extract_score(content, "整体评分")
        if overall == 3.0:
            overall = sum(dims.values()) / len(dims)
        overall_normalized = overall / 5.0

        return FitnessScore(
            role=role,
            overall=overall_normalized,
            dimensions=dims,
            raw_win_rate=win_rate,
            sample_size=len(game_logs),
            details=content[:500],
        )

    @staticmethod
    def _extract_score(text: str, key: str) -> float:
        import re
        pattern = rf"{key}[：:]\s*([\d.]+)"
        match = re.search(pattern, text)
        return float(match.group(1)) if match else 3.0

    @staticmethod
    def _summarize_games(logs: list[dict], role: str) -> str:
        lines = []
        for i, log in enumerate(logs):
            events = log.get("events", [])
            role_events = [e for e in events if e.get("role") == role]
            key_actions = [e for e in role_events if e.get("action") in ("vote", "kill", "check", "save", "poison", "speech", "last_words")]
            lines.append(
                f"对局 {i + 1}（{log.get('winner', '?')}胜，{len([e for e in events if e.get('phase','').startswith('day')])} 天）："
                + "; ".join(f"{a['action']}→{a.get('detail',{}).get('target','?')}" for a in key_actions[:10])
            )
        return "\n".join(lines)


class ConstraintValidator:
    """约束门控器（对标 Hermes ConstraintValidator）。

    确保进化后的 Prompt 不超过安全边界。
    """

    def __init__(
        self,
        max_size_increase: float = 0.3,    # 最大文本膨胀 30%
        max_chars: int = 6000,              # 绝对上限
        min_chars: int = 200,               # 绝对不能低于
        require_structure: bool = True,     # 是否要求保留编号结构
    ) -> None:
        self.max_size_increase = max_size_increase
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.require_structure = require_structure

    def validate(self, original: str, evolved: str) -> ConstraintCheck:
        checks: list[tuple[str, bool, str]] = []

        orig_len = len(original)
        evolved_len = len(evolved)
        growth = (evolved_len - orig_len) / orig_len if orig_len > 0 else 0

        checks.append((
            "尺寸上限",
            evolved_len <= self.max_chars,
            f"{evolved_len}/{self.max_chars} 字符",
        ))
        checks.append((
            "尺寸下限",
            evolved_len >= self.min_chars,
            f"{evolved_len}/{self.min_chars} 字符",
        ))
        checks.append((
            "膨胀控制",
            growth <= self.max_size_increase,
            f"{growth:+.0%}（上限 {self.max_size_increase:+.0%}）",
        ))
        checks.append((
            "内容非空",
            len(evolved.strip()) > 50,
            f"{len(evolved.strip())} 字符有效内容",
        ))

        if self.require_structure:
            has_numbered = any(
                f"{i})" in evolved or f"{i}." in evolved or f"{i}、" in evolved
                for i in range(1, 5)
            )
            checks.append((
                "结构保持",
                has_numbered,
                "包含编号策略条目" if has_numbered else "缺少编号策略结构",
            ))

        all_passed = all(ok for _, ok, _ in checks)
        return ConstraintCheck(passed=all_passed, checks=checks)
