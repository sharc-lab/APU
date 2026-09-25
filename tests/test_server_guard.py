import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import server_guard as sg  # noqa: E402

NETSTAT = """
  Proto  Local Address          Foreign Address        State           PID
  TCP    127.0.0.1:8385         0.0.0.0:0              LISTENING       4242
  TCP    127.0.0.1:8385         127.0.0.1:50521        TIME_WAIT       0
  TCP    [::]:8385              [::]:0                 LISTENING       4243
  TCP    127.0.0.1:18385        0.0.0.0:0              LISTENING       999
  TCP    0.0.0.0:445            0.0.0.0:0              LISTENING       4
"""


def test_parse_netstat_only_listeners_on_exact_port():
    assert sg.parse_netstat_listeners(NETSTAT, 8385) == [4242, 4243]
    assert sg.parse_netstat_listeners(NETSTAT, 445) == [4]
    assert sg.parse_netstat_listeners(NETSTAT, 9999) == []


def test_parse_flags():
    cmd = (r'C:\apu\bin\llama-b10970-cuda\llama-server.exe -m C:\apu\models\q.gguf --port 8385 -c 8192 '
           r'-ctk f16 -ctv f16 -fa on -ngl 99 -np 1 -t 4 --no-context-shift --log-file C:\apu\x.txt')
    f = sg.parse_flags(cmd)
    assert f["ctx"] == "8192" and f["ctk"] == "f16" and f["ctv"] == "f16" and f["fa"] == "on"
    assert f["ngl"] == "99" and f["np"] == "1" and f["threads"] == "4" and f["port"] == "8385"
    assert f["model"] == r"C:\apu\models\q.gguf" and f["no_context_shift"] is True
