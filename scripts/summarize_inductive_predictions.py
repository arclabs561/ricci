#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


@dataclass
class RelationSummary:
    count: int = 0
    reciprocal_rank: float = 0.0
    hits10: int = 0
    rank_sum: int = 0


def rank_queries(graph: Graph, queries: list[Query]) -> list[RankedQuery]:
    ranked = []
    for query in queries:
        scores = dict(query.predictions)
        gold_score = scores[query.target]
        filtered = graph.known[(query.source, query.relation)]

        rank = 1
        candidate_count = 0
        best_corrupt = query.target
        best_corrupt_score = float("-inf")
        for entity, score in query.predictions:
            if entity == query.target or entity in filtered:
                continue
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
            )
        )
    return ranked


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


def hits_at(ranked: list[RankedQuery], k: int) -> float:
    return sum(1 for item in ranked if item.rank <= k) / len(ranked)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize an inductive link-prediction export."
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/fb237_v1_ind"))
    args = parser.parse_args()

    graph = Graph(args.data_dir)
    queries = parse_predictions(args.predictions)
    ranked = rank_queries(graph, queries)
    print_summary(ranked, args.predictions, args.data_dir)


if __name__ == "__main__":
    main()
