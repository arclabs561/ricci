#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections.abc import Callable
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


@dataclass
class RelationDelta:
    count: int = 0
    baseline_rr: float = 0.0
    adjusted_rr: float = 0.0
    baseline_hits10: int = 0
    adjusted_hits10: int = 0
    rank_delta: int = 0
    fixed10: int = 0
    lost10: int = 0


@dataclass(frozen=True)
class SweepPoint:
    alpha: float
    ranked: list[RankedQuery]
    mrr: float
    hits10: float
    median_rank: int
    fixed10: int
    lost10: int


def rank_queries(
    graph: Graph,
    queries: list[Query],
    support_alpha: float = 0.0,
    path_alpha: float = 0.0,
) -> list[RankedQuery]:
    ranked = []
    distance_cache: dict[str, dict[str, int]] = {}
    for query in queries:
        scores = dict(query.predictions)
        distances = None
        if path_alpha != 0.0:
            distances = distances_from_source(graph, distance_cache, query.source)
        gold_score = adjusted_score(
            graph,
            query.relation,
            query.target,
            scores[query.target],
            support_alpha,
            path_alpha,
            distances,
        )
        filtered = graph.known[(query.source, query.relation)]

        rank = 1
        candidate_count = 0
        best_corrupt = query.target
        best_corrupt_score = float("-inf")
        for entity, score in query.predictions:
            if entity == query.target or entity in filtered:
                continue
            score = adjusted_score(
                graph,
                query.relation,
                entity,
                score,
                support_alpha,
                path_alpha,
                distances,
            )
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
    graph: Graph,
    relation: str,
    entity: str,
    score: float,
    support_alpha: float,
    path_alpha: float,
    distances: dict[str, int] | None,
) -> float:
    if support_alpha != 0.0:
        score += support_alpha * math.log1p(graph.relation_support(relation, entity))
    if path_alpha != 0.0 and distances is not None:
        distance = distances.get(entity)
        if distance is not None:
            score += path_alpha / (1.0 + distance)
    return score


def distances_from_source(
    graph: Graph, cache: dict[str, dict[str, int]], source: str, max_hops: int = 5
) -> dict[str, int]:
    if source not in cache:
        cache[source] = bounded_distances(graph, source, max_hops)
    return cache[source]


def bounded_distances(graph: Graph, source: str, max_hops: int) -> dict[str, int]:
    distances = {source: 0}
    frontier = [source]
    for depth in range(1, max_hops + 1):
        next_frontier = []
        for node in frontier:
            for next_node in sorted(graph.adj[node]):
                if next_node in distances:
                    continue
                distances[next_node] = depth
                next_frontier.append(next_node)
        frontier = next_frontier
        if not frontier:
            break
    return distances


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


def print_summary(
    graph: Graph, ranked: list[RankedQuery], predictions: Path, data_dir: Path
) -> None:
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
    print_path_summary(graph, ranked)


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
    print_relation_support_buckets(ranked)


def print_relation_support_buckets(ranked: list[RankedQuery]) -> None:
    print("  full-rank by gold relation-support bucket:")
    print_support_bucket_rows(
        bucket_ranked(ranked, lambda item: support_bucket(item.gold_relation_support))
    )
    print("  full-rank by gold-vs-best-corrupt support:")
    print_support_bucket_rows(
        bucket_ranked(
            ranked,
            lambda item: support_comparison_bucket(
                item.gold_relation_support, item.corrupt_relation_support
            ),
        )
    )


def bucket_ranked(
    ranked: list[RankedQuery], bucket_for: Callable[[RankedQuery], str]
) -> dict[str, RelationSummary]:
    buckets: dict[str, RelationSummary] = defaultdict(RelationSummary)
    for item in ranked:
        summary = buckets[bucket_for(item)]
        summary.count += 1
        summary.reciprocal_rank += 1.0 / item.rank
        summary.hits10 += int(item.rank <= 10)
        summary.rank_sum += item.rank
    return buckets


def support_bucket(support: int) -> str:
    if support == 0:
        return "0"
    if support == 1:
        return "1"
    if support <= 4:
        return "2-4"
    if support <= 9:
        return "5-9"
    return "10+"


def support_comparison_bucket(gold_support: int, corrupt_support: int) -> str:
    if gold_support == corrupt_support:
        return "equal"
    if gold_support > corrupt_support:
        return "gold-higher"
    return "corrupt-higher"


def print_support_bucket_rows(buckets: dict[str, RelationSummary]) -> None:
    order = ["0", "1", "2-4", "5-9", "10+", "gold-higher", "equal", "corrupt-higher"]
    for label in order:
        summary = buckets.get(label)
        if summary is None:
            continue
        print(
            f"    {label}: n {summary.count} "
            f"MRR {summary.reciprocal_rank / summary.count:.3f} "
            f"H@10 {summary.hits10 / summary.count:.3f} "
            f"mean-rank {summary.rank_sum / summary.count:.1f}"
        )


def print_path_summary(graph: Graph, ranked: list[RankedQuery]) -> None:
    gold_buckets: dict[str, RelationSummary] = defaultdict(RelationSummary)
    comparison_buckets: dict[str, RelationSummary] = defaultdict(RelationSummary)
    for item in ranked:
        gold_len = path_len(graph, item.query.source, item.query.target)
        corrupt_len = path_len(graph, item.query.source, item.best_corrupt)
        add_rank(gold_buckets[path_bucket(gold_len)], item)
        add_rank(
            comparison_buckets[path_comparison_bucket(gold_len, corrupt_len)], item
        )

    print("train-path summary:")
    print("  full-rank by gold train-path length:")
    print_path_bucket_rows(gold_buckets)
    print("  full-rank by gold-vs-best-corrupt train path:")
    print_path_bucket_rows(comparison_buckets)


def path_len(graph: Graph, source: str, target: str) -> int | None:
    path = graph.path(source, target)
    if path is None:
        return None
    return len(path) - 1


def path_bucket(length: int | None) -> str:
    if length is None:
        return "none"
    if length <= 3:
        return str(length)
    return "4-5"


def path_comparison_bucket(gold_len: int | None, corrupt_len: int | None) -> str:
    if gold_len is None and corrupt_len is None:
        return "neither"
    if gold_len is None:
        return "only-corrupt"
    if corrupt_len is None:
        return "only-gold"
    if gold_len < corrupt_len:
        return "gold-shorter"
    if gold_len > corrupt_len:
        return "corrupt-shorter"
    return "equal"


def add_rank(summary: RelationSummary, item: RankedQuery) -> None:
    summary.count += 1
    summary.reciprocal_rank += 1.0 / item.rank
    summary.hits10 += int(item.rank <= 10)
    summary.rank_sum += item.rank


def print_path_bucket_rows(buckets: dict[str, RelationSummary]) -> None:
    order = [
        "1",
        "2",
        "3",
        "4-5",
        "none",
        "gold-shorter",
        "equal",
        "corrupt-shorter",
        "only-gold",
        "only-corrupt",
        "neither",
    ]
    for label in order:
        summary = buckets.get(label)
        if summary is None:
            continue
        print(
            f"    {label}: n {summary.count} "
            f"MRR {summary.reciprocal_rank / summary.count:.3f} "
            f"H@10 {summary.hits10 / summary.count:.3f} "
            f"mean-rank {summary.rank_sum / summary.count:.1f}"
        )


def print_path_patterns(graph: Graph, ranked: list[RankedQuery]) -> None:
    print("train-path relation patterns among failures:")
    print_path_pattern_group(graph, ranked, "rank>10", min_rank=11)
    print_path_pattern_group(graph, ranked, "rank>50", min_rank=51)


def print_path_pattern_group(
    graph: Graph, ranked: list[RankedQuery], label: str, min_rank: int
) -> None:
    rows = [item for item in ranked if item.rank >= min_rank]
    patterns: Counter[tuple[str, ...]] = Counter(
        path_pattern(graph, item.query.source, item.query.target) for item in rows
    )
    print(f"  {label}: {len(rows)} cases")
    if not patterns:
        print("    <none>")
        return
    for pattern, count in patterns.most_common(8):
        print(f"    {format_pattern(pattern)}: {count}")


def path_pattern(graph: Graph, source: str, target: str) -> tuple[str, ...]:
    path = graph.labeled_path(source, target)
    if path is None:
        return ("<no-path>",)
    if not path:
        return ("<self>",)
    return tuple(relation for relation, _node in path)


def format_pattern(pattern: tuple[str, ...]) -> str:
    return " -> ".join(pattern)


def print_support_sweep(
    graph: Graph, queries: list[Query], baseline: list[RankedQuery]
) -> None:
    print("relation-support prior sweep:")
    print("  score = model_score + alpha * log1p(train relation support)")
    print("  alpha    MRR    H@10  median  fixed@10  lost@10")
    baseline_hits = [item.rank <= 10 for item in baseline]
    points = []
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
        point = SweepPoint(
            alpha=alpha,
            ranked=ranked,
            mrr=mean_reciprocal_rank(ranked),
            hits10=hits_at(ranked, 10),
            median_rank=ranks[len(ranks) // 2],
            fixed10=fixed,
            lost10=lost,
        )
        points.append(point)
        print(
            f"  {alpha:>4.1f}  {point.mrr:.4f}  "
            f"{point.hits10:.4f}  {point.median_rank:>6}  "
            f"{fixed:>8}  {lost:>7}"
        )
    best_mrr = max(points, key=lambda point: (point.mrr, point.hits10))
    best_hits10 = max(points, key=lambda point: (point.hits10, point.mrr))
    print_support_delta_details(baseline, best_mrr, "best MRR")
    if best_hits10.alpha != best_mrr.alpha:
        print_support_delta_details(baseline, best_hits10, "best H@10")


def print_evidence_sweep(
    graph: Graph, queries: list[Query], baseline: list[RankedQuery]
) -> None:
    print("evidence calibration sweep:")
    print(
        "  score = model_score + support_alpha * log1p(train relation support) "
        "+ path_alpha / (1 + train path length)"
    )
    print("  support  path    MRR    H@10  median  fixed@10  lost@10")
    baseline_hits = [item.rank <= 10 for item in baseline]
    points = []
    for support_alpha in [0.0, 0.4, 0.8]:
        for path_alpha in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
            ranked = rank_queries(
                graph,
                queries,
                support_alpha=support_alpha,
                path_alpha=path_alpha,
            )
            ranks = sorted(item.rank for item in ranked)
            hits = [item.rank <= 10 for item in ranked]
            fixed = sum(
                1 for before, after in zip(baseline_hits, hits) if not before and after
            )
            lost = sum(
                1 for before, after in zip(baseline_hits, hits) if before and not after
            )
            mrr = mean_reciprocal_rank(ranked)
            hits10 = hits_at(ranked, 10)
            points.append(
                (
                    mrr,
                    hits10,
                    support_alpha,
                    path_alpha,
                    ranks[len(ranks) // 2],
                    fixed,
                    lost,
                )
            )
            print(
                f"    {support_alpha:>4.1f}  {path_alpha:>4.2f}  "
                f"{mrr:.4f}  {hits10:.4f}  {ranks[len(ranks) // 2]:>6}  "
                f"{fixed:>8}  {lost:>7}"
            )

    best = max(points, key=lambda row: (row[0], row[1]))
    best_hits10 = max(points, key=lambda row: (row[1], row[0]))
    print(
        "  best MRR: "
        f"support {best[2]:.1f} path {best[3]:.2f} "
        f"MRR {best[0]:.4f} H@10 {best[1]:.4f} "
        f"median {best[4]} fixed@10 {best[5]} lost@10 {best[6]}"
    )
    if best_hits10 != best:
        print(
            "  best H@10: "
            f"support {best_hits10[2]:.1f} path {best_hits10[3]:.2f} "
            f"MRR {best_hits10[0]:.4f} H@10 {best_hits10[1]:.4f} "
            f"median {best_hits10[4]} fixed@10 {best_hits10[5]} "
            f"lost@10 {best_hits10[6]}"
        )


def print_support_delta_details(
    baseline: list[RankedQuery], point: SweepPoint, label: str
) -> None:
    print(f"relation-support prior deltas ({label}, alpha {point.alpha:.1f}):")
    deltas: dict[str, RelationDelta] = defaultdict(RelationDelta)
    for before, after in zip(baseline, point.ranked):
        delta = deltas[before.query.relation]
        delta.count += 1
        delta.baseline_rr += 1.0 / before.rank
        delta.adjusted_rr += 1.0 / after.rank
        delta.baseline_hits10 += int(before.rank <= 10)
        delta.adjusted_hits10 += int(after.rank <= 10)
        delta.rank_delta += after.rank - before.rank
        delta.fixed10 += int(before.rank > 10 and after.rank <= 10)
        delta.lost10 += int(before.rank <= 10 and after.rank > 10)

    def relation_row(row: tuple[str, RelationDelta]) -> tuple[int, float]:
        _relation, delta = row
        rr_delta = delta.adjusted_rr - delta.baseline_rr
        return (delta.fixed10 - delta.lost10, rr_delta)

    helped = sorted(deltas.items(), key=relation_row, reverse=True)
    hurt = sorted(deltas.items(), key=relation_row)
    print("  most helped relations:")
    print_relation_delta_rows(helped[:6])
    print("  most hurt relations:")
    print_relation_delta_rows(hurt[:6])

    fixed = [
        (before, after)
        for before, after in zip(baseline, point.ranked)
        if before.rank > 10 and after.rank <= 10
    ]
    lost = [
        (before, after)
        for before, after in zip(baseline, point.ranked)
        if before.rank <= 10 and after.rank > 10
    ]
    print("  largest fixed@10 cases:")
    print_case_moves(
        sorted(fixed, key=lambda pair: pair[0].rank - pair[1].rank, reverse=True)[:5]
    )
    print("  largest lost@10 cases:")
    print_case_moves(
        sorted(lost, key=lambda pair: pair[1].rank - pair[0].rank, reverse=True)[:5]
    )


def print_relation_delta_rows(rows: list[tuple[str, RelationDelta]]) -> None:
    for relation, delta in rows:
        h10_before = delta.baseline_hits10 / delta.count
        h10_after = delta.adjusted_hits10 / delta.count
        rr_delta = (delta.adjusted_rr - delta.baseline_rr) / delta.count
        mean_rank_delta = delta.rank_delta / delta.count
        print(
            f"    {relation}: n {delta.count} "
            f"fixed {delta.fixed10} lost {delta.lost10} "
            f"H@10 {h10_before:.3f}->{h10_after:.3f} "
            f"MRR_delta {rr_delta:+.4f} "
            f"mean_rank_delta {mean_rank_delta:+.1f}"
        )


def print_case_moves(pairs: list[tuple[RankedQuery, RankedQuery]]) -> None:
    if not pairs:
        print("    <none>")
        return
    for before, after in pairs:
        query = before.query
        print(
            f"    {query.direction} {query.raw_head} {query.raw_relation} {query.raw_tail}: "
            f"rank {before.rank}->{after.rank} "
            f"gold-support {before.gold_relation_support} "
            f"best-corrupt-support {before.corrupt_relation_support}->{after.corrupt_relation_support} "
            f"best-corrupt {before.best_corrupt}->{after.best_corrupt}"
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
    parser.add_argument(
        "--path-patterns",
        action="store_true",
        help="Also print frequent relation-labeled train-path patterns in failures.",
    )
    parser.add_argument(
        "--evidence-sweep",
        action="store_true",
        help="Also sweep a diagnostic relation-support plus train-path calibration.",
    )
    args = parser.parse_args()

    graph = Graph(args.data_dir)
    queries = parse_predictions(args.predictions)
    ranked = rank_queries(graph, queries)
    print_summary(graph, ranked, args.predictions, args.data_dir)
    if args.path_patterns:
        print_path_patterns(graph, ranked)
    if args.support_sweep:
        print_support_sweep(graph, queries, ranked)
    if args.evidence_sweep:
        print_evidence_sweep(graph, queries, ranked)


if __name__ == "__main__":
    main()
