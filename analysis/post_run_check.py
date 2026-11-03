"""Adapter: the post-run check (A, src/dse/post_run_check.py) behind analysis/hybrid_full_pipeline.py's PostRunCheck
interface (make_post_run_check(repo).check(task_record) -> {"escalate", "reason"}).

The check sees exactly what analysis/post_run_check_replay.py gives it: the recorded turn (grader fields already
stripped by the pipeline's strip_grader) plus the session's system prompt, the task's user text and the tools schema,
rebuilt from the seed (post_run_check_replay.sessions). It never reads the grader, the reference calls or the expected
end state; L3's own no-grader test is tests/test_post_run_check.py.
"""
from __future__ import annotations

import sys
from pathlib import Path


class _RecordedPostRunCheck:
    def __init__(self, repo, config=None):
        self.repo, self.config = Path(repo), config
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        sys.path.insert(0, str(self.repo / "analysis"))
        import post_run_check_replay as replay  # noqa: PLC0415
        from src.dse import post_run_check as prc  # noqa: PLC0415
        self.replay, self.prc, self._sessions = replay, prc, {}

    def _session(self, seed):
        if seed not in self._sessions:
            self._sessions.update(self.replay.sessions(self.repo, [seed]))
        return self._sessions[seed]

    def check(self, task_record: dict) -> dict:
        sp, users, schema = self._session(task_record["seed"])
        d = self.prc.check_record(task_record, sp, users[task_record["turn_idx"]], schema, self.config)
        return {"escalate": d["decision"] == "escalate", "reason": ",".join(d["fired"]) or "pass",
                "layers_fired": list(d["fired"])}


def make_post_run_check(repo, config=None):
    """config: the L3 config (src/dse/l3_invariants.CONFIG_V1 when None, as every existing row uses; CONFIG_TUNED
    for the tuned check)."""
    return _RecordedPostRunCheck(repo, config)
