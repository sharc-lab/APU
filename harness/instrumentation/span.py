"""Span accumulator and CategoryMetrics for instrumentation."""

from dataclasses import dataclass, field


@dataclass
class Span:
    """Accumulator for one instrumentation category."""
    cpu_ns: int = 0
    wall_ns: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    count: int = 0

    def record(self, cpu: int, wall: int, b_in: int = 0, b_out: int = 0) -> None:
        """Record a single measurement for this span."""
        self.cpu_ns += cpu
        self.wall_ns += wall
        self.bytes_in += b_in
        self.bytes_out += b_out
        self.count += 1

    def merge(self, other: "Span") -> None:
        """Merge another Span into this one."""
        self.cpu_ns += other.cpu_ns
        self.wall_ns += other.wall_ns
        self.bytes_in += other.bytes_in
        self.bytes_out += other.bytes_out
        self.count += other.count

    def to_dict(self) -> dict:
        """Convert to dict (CategoryMetrics format)."""
        return {
            "cpu_ns": self.cpu_ns,
            "wall_ns": self.wall_ns,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "count": self.count,
        }
