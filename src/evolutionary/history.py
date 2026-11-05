"""Generation snapshots and JSON persistence for evolutionary runs."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

from evolutionary.individual import SearchSpaceEntry

GeneValue: TypeAlias = int | float
IndividualSnapshot: TypeAlias = tuple[str, dict[str, GeneValue], float]


@dataclass(frozen=True)
class GenerationRecord:
    """Immutable snapshot of one evaluated genetic-algorithm generation."""

    generation: int
    individuals: tuple[IndividualSnapshot, ...]
    elite_ids: tuple[str, ...]
    eta_c: float
    best_fitness: float
    seconds: float
    stalled_stop: bool = False


class HistoryRecorder:
    """Collect generation snapshots and persist run metadata as JSON."""

    def __init__(
        self,
        population_size: int,
        generation_budget: int,
        stall_limit: int | None,
        search_space: dict[str, SearchSpaceEntry],
    ):
        self.population_size = population_size
        self.generation_budget = generation_budget
        self.stall_limit = stall_limit
        self.search_space: dict[str, SearchSpaceEntry] = {
            name: {
                "min": entry["min"],
                "max": entry["max"],
                "type": entry["type"],
            }
            for name, entry in search_space.items()
        }
        self.records: list[GenerationRecord] = []

    def __call__(self, record: GenerationRecord) -> None:
        """Append one completed generation snapshot."""
        self.records.append(record)

    def save(self, path: str | Path) -> None:
        """Write run metadata and generation snapshots to a JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        search_space_metadata: dict[str, dict[str, int | float | str]] = {}
        for name, entry in self.search_space.items():
            if entry["type"] is int:
                gene_type = "int"
            elif entry["type"] is float:
                gene_type = "float"
            else:
                raise ValueError(f"Unsupported search-space type for {name!r}.")
            search_space_metadata[name] = {
                "min": entry["min"],
                "max": entry["max"],
                "type": gene_type,
            }
        metadata: dict[str, object] = {
            "population_size": self.population_size,
            "generation_budget": self.generation_budget,
            "stall_limit": self.stall_limit,
            "search_space": search_space_metadata,
        }
        payload = {
            "metadata": metadata,
            "records": [asdict(record) for record in self.records],
        }
        output_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "HistoryRecorder":
        """Load a run history previously written by `save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        search_space: dict[str, SearchSpaceEntry] = {}
        for name, entry in metadata["search_space"].items():
            gene_type: type = int
            if entry["type"] == "int":
                gene_type = int
            elif entry["type"] == "float":
                gene_type = float
            else:
                raise ValueError(f"Unsupported search-space type for {name!r}.")
            search_space[name] = {
                "min": entry["min"],
                "max": entry["max"],
                "type": gene_type,
            }
        recorder = cls(
            population_size=metadata["population_size"],
            generation_budget=metadata["generation_budget"],
            stall_limit=metadata["stall_limit"],
            search_space=search_space,
        )
        for item in payload["records"]:
            individuals = tuple(
                (individual[0], dict(individual[1]), individual[2])
                for individual in item["individuals"]
            )
            recorder(
                GenerationRecord(
                    generation=item["generation"],
                    individuals=individuals,
                    elite_ids=tuple(item["elite_ids"]),
                    eta_c=item["eta_c"],
                    best_fitness=item["best_fitness"],
                    seconds=item["seconds"],
                    stalled_stop=item["stalled_stop"],
                )
            )
        return recorder
