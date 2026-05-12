"""狼人杀 Agent 自进化入口。

用法：
    python -m werewolf.evolve                       # 默认 3 代各 20 局
    python -m werewolf.evolve --generations 5 --games 50
    python -m werewolf.evolve --role 狼人            # 只优化狼人
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 支持直接执行和模块执行两种方式
if __name__ == "__main__" and __package__ is None:
    _parent = Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    __package__ = "werewolf"

from .agents.factory import _ROLE_MAP
from .config import GameConfig, Role
from .evaluation.leaderboard import Leaderboard
from .evolution import StrategyOptimizer, EvolutionTracker
from .evolution.tracker import GenerationMetrics
from .llm import create_llm
from .orchestrator import GameOrchestrator


def run_games(llm, config: GameConfig, n: int, prompt_overrides: dict[str, str] | None = None, verbose: bool = False) -> list[dict]:
    """运行多局游戏，返回日志列表。"""
    logs: list[dict] = []
    for i in range(n):
        if verbose:
            print(f"  对局 {i + 1}/{n}...", end=" ", flush=True)
        orch = GameOrchestrator(llm, config=config, verbose=False)
        state = orch.setup()

        if prompt_overrides:
            for pid, agent in orch.agents.items():
                role = state.get_player(pid).role.value
                if role in prompt_overrides:
                    agent.system_prompt = prompt_overrides[role]

        result = orch.run_game()
        orch.logger.save()
        output_dir = Path(__file__).resolve().parent / "output"
        log_files = sorted(output_dir.glob("game_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if log_files:
            logs.append(json.loads(log_files[0].read_text()))
        if verbose:
            winner = result.winner.value if result.winner else "?"
            print(f"{winner} ({result.total_rounds}天)")

    return logs


def collect_metrics(logs: list[dict]) -> dict[str, dict]:
    """汇总所有对局的指标，按角色聚合。"""
    role_wins: dict[str, int] = {}
    role_games: dict[str, int] = {}
    for log in logs:
        winner = log.get("winner", "")
        events = log.get("events", [])
        roles_seen: set[str] = set()
        for e in events:
            role = e.get("role", "")
            if role and role not in roles_seen:
                roles_seen.add(role)
                role_games[role] = role_games.get(role, 0) + 1
                if (role == "狼人" and winner == "狼人阵营") or (role != "狼人" and winner == "好人阵营"):
                    role_wins[role] = role_wins.get(role, 0) + 1

    result: dict[str, dict] = {}
    for role in role_games:
        games = role_games[role]
        wins = role_wins.get(role, 0)
        result[role] = {"win_rate": wins / games if games > 0 else 0.5, "games_played": games}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="狼人杀 Agent 自进化")
    parser.add_argument("--generations", type=int, default=3, help="进化代数")
    parser.add_argument("--games", type=int, default=20, help="每代对局数")
    parser.add_argument("--role", type=str, default=None, help="只优化指定角色（狼人/预言家/女巫/猎人/白痴/村民）")
    args = parser.parse_args()

    print("=" * 60)
    print("  狼人杀 Agent 自进化系统")
    print("=" * 60)
    print(f"  代数：{args.generations}，每代 {args.games} 局")
    if args.role:
        print(f"  优化目标：{args.role}")
    print()

    llm = create_llm(temperature=0.7)
    optimizer_llm = create_llm(temperature=0.3)
    optimizer = StrategyOptimizer(optimizer_llm)
    tracker = EvolutionTracker()
    config = GameConfig()

    current_prompts: dict[str, str] = {}
    for role in Role:
        cls = _ROLE_MAP.get(role)
        if cls:
            current_prompts[role.value] = cls.system_prompt

    leaderboard = Leaderboard()

    for gen in range(args.generations):
        print(f"{'─' * 50}")
        print(f"  第 {gen + 1}/{args.generations} 代")
        print(f"{'─' * 50}")

        print(f"\n  [1/3] 运行 {args.games} 局对局...")
        logs = run_games(llm, config, args.games, prompt_overrides=current_prompts, verbose=True)

        print(f"\n  [2/3] 分析表现...")
        metrics = collect_metrics(logs)
        for role, m in metrics.items():
            print(f"    {role}: 胜率 {m['win_rate']:.0%} ({m['games_played']} 局)")

        total_games = len(logs)
        wolf_wins = sum(1 for l in logs if l.get("winner") == "狼人阵营")
        good_wins = total_games - wolf_wins

        tracker.record_generation(GenerationMetrics(
            generation=gen + 1,
            games_played=total_games,
            werewolf_wins=wolf_wins,
            good_wins=good_wins,
            avg_rounds=sum(len(l.get("events", [])) for l in logs) / max(total_games, 1),
            role_prompts={k: v[:80] + "..." for k, v in current_prompts.items()},
        ))

        leaderboard.record_game(
            "DeepSeek-V4", f"gen{gen + 1}",
            "狼人阵营" if wolf_wins > good_wins else "好人阵营",
            sum(len(l.get("events", [])) for l in logs) // max(total_games, 1),
        )

        if gen == args.generations - 1:
            break

        print(f"\n  [3/3] 优化 System Prompt...")
        roles_to_optimize = [args.role] if args.role else list(metrics.keys())

        for role in roles_to_optimize:
            if role not in current_prompts:
                continue
            m = metrics.get(role, {"win_rate": 0.5})
            print(f"    分析 {role} 的弱点...")
            weaknesses = optimizer.analyze_weaknesses(role, logs)

            print(f"    生成 {role} 的优化 Prompt...")
            new_prompt = optimizer.optimize(role, current_prompts[role], m, weaknesses)
            current_prompts[role] = new_prompt
            print(f"    {role} Prompt 已更新 ({len(current_prompts[role])} → {len(new_prompt)} 字)")

        print()

    print(f"\n{'=' * 60}")
    print(f"  进化完成！")
    print(f"{'=' * 60}")
    print(f"\n{tracker.improvement_summary()}")
    print(f"\n{leaderboard.report()}")
    print(f"\n进化历史已保存至 werewolf/data/")


if __name__ == "__main__":
    main()
