#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Query:
    raw_head: str
    raw_relation: str
    raw_tail: str
    direction: str
    source: str
    relation: str
    target: str
    predictions: list[tuple[str, float]]


@dataclass(frozen=True)
class Case:
    query: Query
    rank: int
    gold_score: float
    best_corrupt: str
    best_corrupt_score: float


class Graph:
    def __init__(self, data_dir: Path) -> None:
        self.train = self._read(data_dir / "train.txt")
        self.valid = self._read(data_dir / "valid.txt")
        self.test = self._read(data_dir / "test.txt")
        self.all = self.train + self.valid + self.test

        self.out: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.inc: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.adj: dict[str, set[str]] = defaultdict(set)
        self.known: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.relation_tails: dict[str, Counter[str]] = defaultdict(Counter)
        self.relation_heads: dict[str, Counter[str]] = defaultdict(Counter)

        for head, relation, tail in self.train:
            self.out[head].append((relation, tail))
            self.inc[tail].append((relation, head))
            self.adj[head].add(tail)
            self.adj[tail].add(head)
            self.relation_tails[relation][tail] += 1
            self.relation_heads[relation][head] += 1

        for head, relation, tail in self.all:
            self.known[(head, relation)].add(tail)
            self.known[(tail, inverse_relation(relation))].add(head)

    @staticmethod
    def _read(path: Path) -> list[tuple[str, str, str]]:
        triples = []
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{lineno}: expected 3 tab-separated fields")
            triples.append((parts[0], parts[1], parts[2]))
        return triples

    def signature(self, entity: str) -> Counter[str]:
        signature = Counter()
        for relation, _tail in self.out[entity]:
            signature[f"out {relation}"] += 1
        for relation, _head in self.inc[entity]:
            signature[f"in {relation}"] += 1
        return signature

    def path(self, source: str, target: str, max_hops: int = 5) -> list[str] | None:
        if source == target:
            return [source]
        seen = {source}
        queue: deque[tuple[str, list[str]]] = deque([(source, [source])])
        while queue:
            node, path = queue.popleft()
            if len(path) > max_hops:
                continue
            for next_node in sorted(self.adj[node]):
                if next_node in seen:
                    continue
                next_path = [*path, next_node]
                if next_node == target:
                    return next_path
                seen.add(next_node)
                queue.append((next_node, next_path))
        return None

    def relation_support(self, relation: str, entity: str) -> int:
        if relation.endswith("^-1"):
            return self.relation_heads[inverse_relation(relation)][entity]
        return self.relation_tails[relation][entity]


def inverse_relation(relation: str) -> str:
    if relation.endswith("^-1"):
        return relation[:-3]
    return f"{relation}^-1"


def parse_predictions(path: Path) -> list[Query]:
    lines = path.read_text().splitlines()
    if len(lines) % 3 != 0:
        raise ValueError(f"{path}: expected 3 lines per query")

    queries: list[Query] = []
    for i in range(0, len(lines), 3):
        head, relation, tail = lines[i].split(" ", 2)
        head_predictions = parse_prediction_line(lines[i + 1], "Heads")
        tail_predictions = parse_prediction_line(lines[i + 2], "Tails")
        queries.append(
            Query(
                raw_head=head,
                raw_relation=relation,
                raw_tail=tail,
                direction="tail",
                source=head,
                relation=relation,
                target=tail,
                predictions=tail_predictions,
            )
        )
        queries.append(
            Query(
                raw_head=head,
                raw_relation=relation,
                raw_tail=tail,
                direction="head",
                source=tail,
                relation=inverse_relation(relation),
                target=head,
                predictions=head_predictions,
            )
        )
    return queries


def parse_prediction_line(line: str, label: str) -> list[tuple[str, float]]:
    parts = line.split("\t")
    if parts[0] != f"{label}:":
        raise ValueError(f"expected {label}: line, got {parts[0]!r}")
    if (len(parts) - 1) % 2 != 0:
        raise ValueError(f"malformed {label}: prediction line")
    return [(parts[i], float(parts[i + 1])) for i in range(1, len(parts), 2)]


def rank_case(graph: Graph, query: Query) -> Case:
    scores = dict(query.predictions)
    gold_score = scores[query.target]
    filtered = graph.known[(query.source, query.relation)]

    rank = 1
    best_corrupt = query.target
    best_corrupt_score = float("-inf")
    for entity, score in query.predictions:
        if entity == query.target or entity in filtered:
            continue
        if score > gold_score:
            rank += 1
        if score > best_corrupt_score:
            best_corrupt = entity
            best_corrupt_score = score

    return Case(query, rank, gold_score, best_corrupt, best_corrupt_score)


def choose_cases(cases: list[Case], args: argparse.Namespace) -> list[Case]:
    if args.case:
        head, relation, tail = args.case
        matches = [
            case
            for case in cases
            if case.query.raw_head == head
            and case.query.raw_relation == relation
            and case.query.raw_tail == tail
            and case.query.direction == args.direction
        ]
        if not matches:
            raise SystemExit("case not found in prediction export")
        return matches

    return sorted(cases, key=lambda case: (-case.rank, case.gold_score))[: args.worst]


def print_case(graph: Graph, case: Case, top_k: int) -> None:
    query = case.query
    margin = case.gold_score - case.best_corrupt_score
    print()
    print(
        f"{query.direction} query: {query.raw_head} {query.raw_relation} {query.raw_tail}"
    )
    print(
        f"scored as: source={query.source} relation={query.relation} target={query.target}"
    )
    print(
        f"rank={case.rank} gold={case.gold_score:.6f} "
        f"best_corrupt={case.best_corrupt} corrupt_score={case.best_corrupt_score:.6f} "
        f"margin={margin:.6f}"
    )

    gold_path = graph.path(query.source, query.target)
    corrupt_path = graph.path(query.source, case.best_corrupt)
    print(f"train path to gold: {format_path(gold_path)}")
    print(f"train path to best corrupt: {format_path(corrupt_path)}")
    print(
        "observed train adjacency: "
        f"gold={str(query.target in graph.adj[query.source]).lower()} "
        f"best_corrupt={str(case.best_corrupt in graph.adj[query.source]).lower()}"
    )
    print(
        "train relation support: "
        f"gold={graph.relation_support(query.relation, query.target)} "
        f"best_corrupt={graph.relation_support(query.relation, case.best_corrupt)}"
    )

    print_relation_context(graph, query.relation)
    print_entity_context(graph, "source", query.source)
    print_entity_context(graph, "gold", query.target)
    print_entity_context(graph, "best corrupt", case.best_corrupt)

    print("top predictions:")
    for rank, (entity, score) in enumerate(query.predictions[:top_k], 1):
        marker = " gold" if entity == query.target else ""
        signature = ", ".join(
            f"{name} ({count})"
            for name, count in graph.signature(entity).most_common(3)
        )
        support = graph.relation_support(query.relation, entity)
        print(
            f"  {rank:2d}. {entity:12s} {score:9.6f}{marker}  "
            f"rel-support={support}  {signature}"
        )


def print_relation_context(graph: Graph, relation: str) -> None:
    base_relation = inverse_relation(relation) if relation.endswith("^-1") else relation
    tails = graph.relation_tails[base_relation]
    heads = graph.relation_heads[base_relation]
    print(f"train relation {base_relation}: {sum(tails.values())} triples")
    print("  top tails: " + format_counter(tails, 5))
    print("  top heads: " + format_counter(heads, 5))


def print_entity_context(graph: Graph, label: str, entity: str) -> None:
    out_counts = Counter(relation for relation, _tail in graph.out[entity])
    in_counts = Counter(relation for relation, _head in graph.inc[entity])
    print(f"{label} {entity}: out={len(graph.out[entity])} in={len(graph.inc[entity])}")
    if out_counts:
        print("  out: " + format_counter(out_counts, 4))
    if in_counts:
        print("  in:  " + format_counter(in_counts, 4))


def format_counter(counter: Counter[str], limit: int) -> str:
    if not counter:
        return "<none>"
    return ", ".join(f"{key} ({count})" for key, count in counter.most_common(limit))


def format_path(path: list[str] | None) -> str:
    if path is None:
        return "<none within 5 hops>"
    return " -> ".join(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect concrete inductive link-prediction cases."
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/fb237_v1_ind"))
    parser.add_argument("--worst", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--case", nargs=3, metavar=("HEAD", "RELATION", "TAIL"))
    parser.add_argument("--direction", choices=("head", "tail"), default="tail")
    args = parser.parse_args()

    graph = Graph(args.data_dir)
    queries = parse_predictions(args.predictions)
    cases = [rank_case(graph, query) for query in queries]
    selected = choose_cases(cases, args)

    print(f"prediction export: {args.predictions}")
    print(f"data dir: {args.data_dir}")
    print(f"queries: {len(queries)}")
    for case in selected:
        print_case(graph, case, args.top_k)


if __name__ == "__main__":
    main()
