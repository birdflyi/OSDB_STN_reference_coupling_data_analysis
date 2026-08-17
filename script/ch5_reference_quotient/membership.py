"""Artifact-to-project membership parsing and quotient invariants."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable, Optional


PROJECT_MEMBERSHIP = re.compile(r"R_(\d+)")
PLACEHOLDER_IDENTITY = re.compile(r"(?:^|_)None(?:#|$)", re.IGNORECASE)


def project_memberships(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    matches = PROJECT_MEMBERSHIP.findall(str(value))
    return tuple(dict.fromkeys(matches))


def unique_project_membership(value: object) -> Optional[str]:
    memberships = project_memberships(value)
    return memberships[0] if len(memberships) == 1 else None


def normalized_entity_identity(value: object) -> Optional[str]:
    if value is None:
        return None
    identity = str(value).strip()
    if not identity or identity.lower() == "nan" or PLACEHOLDER_IDENTITY.search(identity):
        return None
    return identity


def canonical_project_entity_identity(entity_id: object, aggregate_id: object) -> Optional[str]:
    identity = normalized_entity_identity(entity_id)
    if identity is not None:
        return identity
    project_id = unique_project_membership(aggregate_id)
    return f"R_{project_id}" if project_id is not None else None


def classify_membership(value: object, aggregate_type: object) -> str:
    memberships = project_memberships(value)
    if len(memberships) == 1:
        return "project_mappable"
    if len(memberships) > 1:
        return "ambiguous"
    if str(aggregate_type) == "Object":
        return "non_project"
    return "unresolved"


class MembershipRegistry:
    """Disk-backed audit of the single-valued membership function pi."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS memberships (entity_id TEXT NOT NULL, project_id TEXT NOT NULL, "
            "PRIMARY KEY(entity_id, project_id)) WITHOUT ROWID"
        )

    def add(self, pairs: Iterable[tuple[str, str]]) -> None:
        self.connection.executemany("INSERT OR IGNORE INTO memberships VALUES (?, ?)", pairs)

    def commit(self) -> None:
        self.connection.commit()

    def summary(self) -> dict[str, int]:
        self.commit()
        unique_entities = self.connection.execute(
            "SELECT COUNT(DISTINCT entity_id) FROM memberships"
        ).fetchone()[0]
        conflicts = self.connection.execute(
            "SELECT COUNT(*) FROM (SELECT entity_id FROM memberships GROUP BY entity_id "
            "HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        max_memberships = self.connection.execute(
            "SELECT COALESCE(MAX(n), 0) FROM (SELECT COUNT(*) AS n FROM memberships GROUP BY entity_id)"
        ).fetchone()[0]
        return {
            "unique_project_mappable_entities": int(unique_entities),
            "membership_conflict_entities": int(conflicts),
            "retained_single_membership_entities": int(unique_entities - conflicts),
            "maximum_memberships_per_entity": int(max_memberships),
            "maximum_memberships_per_retained_entity": 1 if unique_entities > conflicts else 0,
        }

    def conflicting_entities(self) -> set[str]:
        rows = self.connection.execute(
            "SELECT entity_id FROM memberships GROUP BY entity_id HAVING COUNT(*) > 1"
        ).fetchall()
        return {str(row[0]) for row in rows}

    def close(self) -> None:
        self.connection.close()
