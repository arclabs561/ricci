#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from inspect_inductive_predictions import Graph, Query, parse_predictions


@dataclass(frozen=True)
class RankedQuery:
    query: Query
    rank: int
    candidate_count: int
    best_corrupt: str
    margin: float
    gold_relation_support: int
    corrupt_relation_support: int


@dataclass
class RelationSummary:
    count: int = 0
    reciprocal_rank: float = 0.0
    hits10: int = 0
    rank_sum: int = 0


def rank_queries(
    graph: Graph, queries: list[Query], support_alpha: float = 0.0
) -> list[RankedQuery]:
    ranked = []
    for query in queries:
        scores = dict(query.predictions)
        gold_score = adjusted_score(
            graph, query.relation, query.target, scores[query.target], support_alpha
        )
        filtered = graph.known[(query.source, query.relation)]

        rank = 1
        candidate_count = 0
        best_corrupt = query.target
        best_corrupt_score = float("-inf")
        for entity, score in query.predictions:
            if entity == query.target or entity in filtered:
                continue
            score = adjusted_score(graph, query.relation, entity, score, support_alpha)
            candidate_count += 1
            if score > gold_score:
                rank += 1
            if score > best_corrupt_score:
                best_corrupt = entity
                best_corrupt_score = score
        ranked.append(
            RankedQuery(
                query=query,
                rank=rank,
                candidate_count=candidate_count,
                best_corrupt=best_corrupt,
                margin=gold_score - best_corrupt_score,
                gold_relation_support=graph.relation_support(
                    query.relation, query.target
                ),
                corrupt_relation_support=graph.relation_support(
                    query.relation, best_corrupt
                ),
            )
        )
    return ranked


def adjusted_score(
    graph: Graph, relation: str, entity: str, score: float, support_alpha: float
) -> float:
    if support_alpha == 0.0:
        return score
    return score + support_alpha * math.log1p(graph.relation_support(relation, entity))


def estimated_hits_at_k(
    rank: int, candidate_count: int, k: int = 10, samples: int = 50
) -> float:
    if candidate_count == 0:
        return 1.0
    false_positive_rate = max(0.0, min(1.0, (rank - 1) / candidate_count))
    score = 0.0
    for false_positives in range(k):
        score += (
            combination(samples, false_positives)
            * false_positive_rate**false_positives
            * (1.0 - false_positive_rate) ** (samples - false_positives)
        )
    return score


def combination(n: int, k: int) -> float:
    if k > n:
        return 0.0
    k = min(k, n - k)
    value = 1.0
    for i in range(k):
        value *= (n - i) / (i + 1)
    return value


def print_summary(ranked: list[RankedQuery], predictions: Path, data_dir: Path) -> None:
    ranks = sorted(item.rank for item in ranked)
    n = len(ranked)
    reciprocal_rank = sum(1.0 / item.rank for item in ranked)
    estimated_hits10 = sum(
        estimated_hits_at_k(item.rank, item.candidate_count) for item in ranked
    )

    print(f"prediction export: {predictions}")
    print(f"data dir: {data_dir}")
    print(f"queries: {n}")
    print(
        "full-ranking filtered: "
        f"MRR {reciprocal_rank / n:.4f} "
        f"H@1 {hits_at(ranked, 1):.4f} "
        f"H@3 {hits_at(ranked, 3):.4f} "
        f"H@10 {hits_at(ranked, 10):.4f} "
        f"H@50 {hits_at(ranked, 50):.4f}"
    )
    print(f"estimated-50 Hits@10 (TorchDrug h@10_50): {estimated_hits10 / n:.4f}")
    print(
        "full rank: "
        f"mean {sum(ranks) / n:.1f} "
        f"median {ranks[n // 2]} "
        f"p90 {ranks[n * 9 // 10]} "
        f"p95 {ranks[n * 95 // 100]} "
        f"p99 {ranks[n * 99 // 100]} "
        f"max {ranks[-1]}"
    )

    relation_stats: dict[str, RelationSummary] = defaultdict(RelationSummary)
    corrupt_counts: Counter[str] = Counter()
    for item in ranked:
        summary = relation_stats[item.query.relation]
        summary.count += 1
        summary.reciprocal_rank += 1.0 / item.rank
        summary.hits10 += int(item.rank <= 10)
        summary.rank_sum += item.rank
        corrupt_counts[item.best_corrupt] += 1

    print("worst relations by full-rank MRR:")
    rows = sorted(
        relation_stats.items(),
        key=lambda row: row[1].reciprocal_rank / row[1].count,
    )
    for relation, summary in rows[:8]:
        print(
            f"  {relation}: n {summary.count} "
            f"MRR {summary.reciprocal_rank / summary.count:.3f} "
            f"H@10 {summary.hits10 / summary.count:.3f} "
            f"mean-rank {summary.rank_sum / summary.count:.1f}"
        )

    print("most frequent best corrupt entities:")
    for entity, count in corrupt_counts.most_common(8):
        print(f"  {entity}: n {count} ({100.0 * count / n:.1f}%)")

    print_relation_support_summary(ranked)


def hits_at(ranked: list[RankedQuery], k: int) -> float:
    return sum(1 for item in ranked if item.rank <= k) / len(ranked)


def print_relation_support_summary(ranked: list[RankedQuery]) -> None:
    supported_gold_zero_corrupt = [
        item
        for item in ranked
        if item.gold_relation_support > 0 and item.corrupt_relation_support == 0
    ]
    support_advantage_losses = [
        item
        for item in ranked
        if item.rank > 10 and item.gold_relation_support > item.corrupt_relation_support
    ]
    zero_zero = [
        item
        for item in ranked
        if item.gold_relation_support == 0 and item.corrupt_relation_support == 0
    ]
    n = len(ranked)
    print("relation-support summary:")
    print(
        "  gold supported, best corrupt unsupported: "
        f"{len(supported_gold_zero_corrupt)} ({100.0 * len(supported_gold_zero_corrupt) / n:.1f}%)"
    )
    print(
        "  same pattern among rank>50 failures: "
        f"{sum(1 for item in supported_gold_zero_corrupt if item.rank > 50)}"
    )
    print(
        "  rank>10 despite higher gold support: "
        f"{len(support_advantage_losses)} ({100.0 * len(support_advantage_losses) / n:.1f}%)"
    )
    print(f"  no train support for either gold or best corrupt: {len(zero_zero)}")

    by_relation = Counter(item.query.relation for item in supported_gold_zero_corrupt)
    if by_relation:
        print("  top supported-gold/unsupported-corrupt relations:")
        for relation, count in by_relation.most_common(8):
            print(f"    {relation}: {count}")


def print_support_sweep(
    graph: Graph, queries: list[Query], baseline: list[RankedQuery]
) -> None:
    print("relation-support prior sweep:")
    print("  score = model_score + alpha * log1p(train relation support)")
    print("  alpha    MRR    H@10  median  fixed@10  lost@10")
    baseline_hits = [item.rank <= 10 for item in baseline]
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        ranked = rank_queries(graph, queries, support_alpha=alpha)
        ranks = sorted(item.rank for item in ranked)
        hits = [item.rank <= 10 for item in ranked]
        fixed = sum(
            1 for before, after in zip(baseline_hits, hits) if not before and after
        )
        lost = sum(
            1 for before, after in zip(baseline_hits, hits) if before and not after
        )
        print(
            f"  {alpha:>4.1f}  {mean_reciprocal_rank(ranked):.4f}  "
            f"{hits_at(ranked, 10):.4f}  {ranks[len(ranks) // 2]:>6}  "
            f"{fixed:>8}  {lost:>7}"
        )


def mean_reciprocal_rank(ranked: list[RankedQuery]) -> float:
    return sum(1.0 / item.rank for item in ranked) / len(ranked)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize an inductive link-prediction export."
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/fb237_v1_ind"))
    parser.add_argument(
        "--support-sweep",
        action="store_true",
        help="Also sweep a diagnostic relation-support prior over exported scores.",
    )
    args = parser.parse_args()

    graph = Graph(args.data_dir)
    queries = parse_predictions(args.predictions)
    ranked = rank_queries(graph, queries)
    print_summary(ranked, args.predictions, args.data_dir)
    if args.support_sweep:
        print_support_sweep(graph, queries, ranked)


if __name__ == "__main__":
    main()
