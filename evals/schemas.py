from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AssertionSpec:
    id: str
    kind: str
    values: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssertionSpec":
        return cls(
            id=data["id"],
            kind=data["kind"],
            values=list(data.get("values", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalSpec:
    eval_id: str
    prompt: str
    assertions: list[AssertionSpec]
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSpec":
        return cls(
            eval_id=data["eval_id"],
            prompt=data["prompt"],
            assertions=[AssertionSpec.from_dict(item) for item in data.get("assertions", [])],
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "prompt": self.prompt,
            "assertions": [assertion.to_dict() for assertion in self.assertions],
            "notes": self.notes,
        }
