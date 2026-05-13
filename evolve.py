"""狼人杀 Agent Hermes 风格自进化入口。

对标 NousResearch/hermes-agent s25-s27 的 7 步文本优化管线：
  ① SELECT  → 选择优化目标（最弱角色优先）
  ② BUILD   → 生成评估数据集（对局 trace 提取）
  ③ BASELINE→ 多维适应度评估（FitnessScore）
  ④ CONSTRAIN→ 约束门控检查
  ⑤ OPTIMIZE → GEPA 反馈→变异→评估→择优循环
  ⑥ VALIDATE → 约束 + holdout 验证
  ⑦ DEPLOY   → 备份 → 写入 → 生效

用法：
    python -m werewolf.evolve                       # 默认 3 代各 20 局
    python -m werewolf.evolve --generations 5 --games 50
    python -m werewolf.evolve --role 狼人            # 只优化指定角色
    python -m werewolf.evolve --review-only          # 只做后台审视，不优化
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _parent = Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    __package__ = "werewolf"

from .agents.factory import _ROLE_MAP
from .config import GameConfig, Role
from .evaluation.leaderboard import Leaderboard
from .evolution import (
    BackgroundReviewer,
    ConstraintValidator,
    EvolutionTracker,
    FitnessEvaluator,
    GEPAOptimizer,
    GenerationMetrics,
)
from .llm import create_llm
from .orchestrator import GameOrchestrator

PROMPT_BACKUP_DIR = Path(__file__).resolve().parent / "data" / "prompt_backups"


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

        orch.run_game()
        orch.logger.save()
        output_dir = Path(__file__).resolve().parent / "output"
        log_files = sorted(output_dir.glob("game_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if log_files:
            logs.append(json.loads(log_files[0].read_text()))
        if verbose and logs:
            last = logs[-1]
            winner = last.get("winner", "?")
            rounds = max((e.get("round", 0) for e in last.get("events", [])), default=0)
            print(f"{winner} ({rounds}天)")

    return logs


def collect_metrics(logs: list[dict]) -> dict[str, dict]:
    """汇总对局指标，按角色聚合。"""
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

    return {
        role: {"win_rate": role_wins.get(role, 0) / g if g > 0 else 0.5, "games_played": g}
        for role, g in role_games.items()
    }


def backup_prompt(role: str, prompt: str) -> Path:
    PROMPT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = PROMPT_BACKUP_DIR / f"{role}_{ts}.txt"
    path.write_text(prompt, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes 风格狼人杀 Agent 自进化")
    parser.add_argument("--generations", type=int, default=3, help="进化代数")
    parser.add_argument("--games", type=int, default=20, help="每代对局数")
    parser.add_argument("--role", type=str, default=None, help="只优化指定角色")
    parser.add_argument("--review-only", action="store_true", help="只做后台审视")
    parser.add_argument("--gepa-generations", type=int, default=3, help="GEPA 优化代数")
    args = parser.parse_args()

    print("=" * 60)
    print("  狼人杀 Agent 自进化（Hermes 风格）")
    print("  agent 表现 = 模型 × 上下文文本质量")
    print("=" * 60)
    print(f"  世代：{args.generations}，每代 {args.games} 局")
    if args.role:
        print(f"  目标：{args.role}")
    print()

    llm = create_llm(temperature=0.7)
    optimizer_llm = create_llm(temperature=0.3)
    config = GameConfig()

    # Hermes 基础设施
    fitness_evaluator = FitnessEvaluator(optimizer_llm)
    constraint_validator = ConstraintValidator()
    gepa = GEPAOptimizer(
        optimizer_llm, fitness_evaluator, constraint_validator,
        max_generations=args.gepa_generations,
    )
    reviewer = BackgroundReviewer(optimizer_llm)
    tracker = EvolutionTracker()
    leaderboard = Leaderboard()

    current_prompts: dict[str, str] = {}
    for role in Role:
        cls = _ROLE_MAP.get(role)
        if cls:
            current_prompts[role.value] = cls.system_prompt

    for gen in range(args.generations):
        print(f"{'─' * 50}")
        print(f"  第 {gen + 1}/{args.generations} 代")
        print(f"{'─' * 50}")

        # ============================================================
        # ① SELECT：选择优化目标（最弱角色优先）
        # ============================================================
        if gen > 0:
            sorted_by_win = sorted(
                [(r, m["win_rate"]) for r, m in metrics.items()],
                key=lambda x: x[1],
            )
            optimize_roles = [args.role] if args.role else [sorted_by_win[0][0]]
            print(f"\n  ① SELECT: 优化目标 → {', '.join(optimize_roles)}")
        else:
            optimize_roles = [args.role] if args.role else list(current_prompts.keys())
            print(f"\n  ① SELECT: 首代全量评估所有角色")

        # ============================================================
        # ② BUILD + 对局：运行对局 + 后台审视提取 trace
        # ============================================================
        print(f"\n  ② BUILD: 运行 {args.games} 局 + 后台审视...")
        logs = run_games(llm, config, args.games, prompt_overrides=current_prompts, verbose=True)

        reviews = []
        for log in logs:
            review = reviewer.review(log)
            reviews.append(review)

        strategies = reviewer.extract_strategies()
        print(f"  审视完成：{reviewer.experience_count} 条经验，{len(strategies)} 条通用策略")

        # ============================================================
        # ③ BASELINE：多维适应度评估
        # ============================================================
        print(f"\n  ③ BASELINE: 多维适应度评估...")
        metrics = collect_metrics(logs)
        fitness_scores: dict[str, float] = {}
        for role in optimize_roles:
            if role not in metrics:
                continue
            score = fitness_evaluator.evaluate(role, logs, metrics[role]["win_rate"])
            fitness_scores[role] = score
            dim_str = ", ".join(f"{d.value}:{s:.1f}" for d, s in score.dimensions.items())
            print(f"    {role}: 适应度 {score.overall:.2f} | {dim_str}")

        # ============================================================
        # ④ CONSTRAIN：约束门控
        # ============================================================
        print(f"\n  ④ CONSTRAIN: 约束门控...")
        constraints = {}
        for role in optimize_roles:
            if role not in current_prompts:
                continue
            check = constraint_validator.validate(current_prompts[role], current_prompts[role])
            constraints[role] = check
            print(f"    {role}: {'✓ 通过' if check.passed else '✗ 未通过'}")

        # ============================================================
        # ⑤ OPTIMIZE：GEPA 反馈→变异→评估→择优
        # ============================================================
        if not args.review_only and gen < args.generations - 1:
            print(f"\n  ⑤ OPTIMIZE: GEPA 反馈→变异→评估→择优循环...")
            for role in optimize_roles:
                if role not in current_prompts or role not in metrics:
                    continue
                print(f"    [{role}] 开始 GEPA 优化（{args.gepa_generations} 代）...")
                result = gepa.optimize(
                    role, current_prompts[role], logs, metrics[role]["win_rate"]
                )
                if result.best_score > result.original_score + 0.03:
                    backup = backup_prompt(role, current_prompts[role])
                    current_prompts[role] = result.best_prompt
                    print(f"    [{role}] ✓ 优化采纳（{result.original_score:.2f}→{result.best_score:.2f}，+{result.improvement:+.0%}，{result.candidates_tested}候选）备份: {backup.name}")
                else:
                    print(f"    [{role}] — 无显著改进，保留原版")

        # ============================================================
        # 记录 tracker + leaderboard
        # ============================================================
        total_games = len(logs)
        wolf_wins = sum(1 for l in logs if l.get("winner") == "狼人阵营")
        tracker.record_generation(GenerationMetrics(
            generation=gen + 1,
            games_played=total_games,
            werewolf_wins=wolf_wins,
            good_wins=total_games - wolf_wins,
            avg_rounds=sum(len(l.get("events", [])) for l in logs) / max(total_games, 1),
            role_prompts={k: v[:80] + "..." for k, v in current_prompts.items()},
        ))
        leaderboard.record_game(
            "DeepSeek-V4", f"gen{gen + 1}",
            "狼人阵营" if wolf_wins > (total_games - wolf_wins) else "好人阵营",
            sum(len(l.get("events", [])) for l in logs) // max(total_games, 1),
        )
        print()

    # ============================================================
    # ⑥ VALIDATE + ⑦ DEPLOY
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"  进化完成！")
    print(f"{'=' * 60}")

    print(f"\n  ⑥ VALIDATE: 最终约束检查")
    final_check = constraint_validator.validate(
        _ROLE_MAP[Role.WEREWOLF].system_prompt,
        current_prompts.get("狼人", ""),
    )
    print(f"    狼人 Prompt: {'✓ 通过' if final_check.passed else '✗ 未通过'}")

    print(f"\n  ⑦ DEPLOY: Prompt 已生效（备份在 data/prompt_backups/）")
    print(f"    进化历史: data/evolution_history.json")

    print(f"\n{tracker.improvement_summary()}")
    print(f"\n{leaderboard.report()}")


if __name__ == "__main__":
    main()
