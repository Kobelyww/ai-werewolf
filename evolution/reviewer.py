"""后台审视器（对标 Hermes s20 BackgroundReviewer）。

每局对战结束后，自动生成审视报告：提取关键经验、更新角色记忆。
这些"经验记忆"作为后续优化时的输入，为 GEPA 优化器提供高质量的 trace。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

REVIEW_PROMPT = """你是狼人杀策略分析师。复盘以下对局，提取可复用的策略经验。

## 对局结果
- 胜者：{winner}
- 总轮数：{rounds} 天

## 关键事件
{key_events}

## 分析任务
1. **胜负关键点**（1-2 句话）：这局输赢的核心原因是什么？
2. **成功策略**（2-3 条）：哪些决策做对了？可以提炼为什么经验？
3. **失败教训**（2-3 条）：哪些决策导致失败？如何避免？
4. **角色表现亮点**：哪个角色表现最好？具体做了什么？
5. **改进建议**（按角色）：每个角色最需要改进的一点

请简洁输出，每项不超过 2 行。"""

STRATEGY_EXTRACT_PROMPT = """从以下多局复盘经验中，提取 5-8 条通用的狼人杀策略原则。

{experiences}

输出格式（每条一行）：
- [策略类型：伪装/推理/协作/时机] 具体策略描述"""


@dataclass
class GameReview:
    game_id: str
    winner: str
    rounds: int
    key_insight: str         # 胜负关键点
    good_strategies: list[str]  # 成功策略
    bad_strategies: list[str]   # 失败教训
    best_player: str = ""       # 表现最佳角色
    improvements: dict[str, str] = field(default_factory=dict)  # role → suggestion

    def format_for_optimizer(self) -> str:
        """格式化为优化器可用的 trace。"""
        parts = [
            f"胜负关键点：{self.key_insight}",
            f"成功策略：{'；'.join(self.good_strategies)}",
            f"失败教训：{'；'.join(self.bad_strategies)}",
        ]
        if self.improvements:
            for role, suggestion in self.improvements.items():
                parts.append(f"[{role}] {suggestion}")
        return "\n".join(parts)


class BackgroundReviewer:
    """后台对局审视器。"""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self._experience_bank: list[str] = []  # 跨对局经验积累

    def review(self, game_log: dict) -> GameReview:
        """对一局进行审视，提取经验。"""
        events = game_log.get("events", [])
        winner = game_log.get("winner", "未知")
        rounds = max((e.get("round", 0) for e in events), default=0)

        key_events = self._format_key_events(events)
        prompt = REVIEW_PROMPT.format(winner=winner, rounds=rounds, key_events=key_events)
        result = self.llm.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)
        if isinstance(content, list):
            content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)

        review = self._parse_review(content, winner, rounds)
        self._experience_bank.append(review.format_for_optimizer())

        if len(self._experience_bank) > 50:
            self._experience_bank = self._experience_bank[-50:]

        return review

    def extract_strategies(self) -> list[str]:
        """从积累的经验中提取通用策略。"""
        if len(self._experience_bank) < 3:
            return []
        prompt = STRATEGY_EXTRACT_PROMPT.format(
            experiences="\n\n---\n\n".join(self._experience_bank[-20:])
        )
        result = self.llm.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)
        if isinstance(content, list):
            content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
        return [l.strip("- ").strip() for l in content.split("\n") if l.strip().startswith("-")]

    @property
    def experience_count(self) -> int:
        return len(self._experience_bank)

    def clear(self) -> None:
        self._experience_bank.clear()

    @staticmethod
    def _format_key_events(events: list[dict]) -> str:
        key_types = {"save", "poison", "check", "kill", "vote", "hunter_shoot", "speech"}
        key = [e for e in events if e.get("action") in key_types]
        return "\n".join(
            f"第{e.get('round', 0)}天 [{e.get('phase', '')}] {e.get('actor', '?')}号({e.get('role', '')})"
            f" {e.get('action')} → {e.get('detail', {}).get('target', '')}"
            for e in key[-30:]
        )

    @staticmethod
    def _parse_review(content: str, winner: str, rounds: int) -> GameReview:
        import re

        def extract_section(label: str) -> list[str]:
            pattern = rf"{label}[：:]\s*\n?(.+?)(?=\n\s*\d\.|\n\s*$|\Z)"
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                return []
            return [l.strip("- ").strip() for l in match.group(1).strip().split("\n") if l.strip().startswith("-")]

        insight_match = re.search(r"胜负关键点[：:]\s*(.+?)(?=\n\s*\d\.|\n\n|\Z)", content, re.DOTALL)
        insight = insight_match.group(1).strip() if insight_match else ""

        return GameReview(
            game_id="",
            winner=winner,
            rounds=rounds,
            key_insight=insight,
            good_strategies=extract_section("成功策略"),
            bad_strategies=extract_section("失败教训"),
        )
