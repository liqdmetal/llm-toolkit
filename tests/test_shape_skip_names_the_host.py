"""A shaping skip must say WHICH host refused it.

Why this exists
---------------
The shaping lane's whole job is to be a soft dependency: when it cannot run, the
caller uses its own prompt. The skip message is therefore the only evidence an
operator ever sees, and it has to be actionable. MEASURED 2026-11-28, the message
that fired 100% of the time did not name a host at all:

    SHAPE_SKIP: model 'defiant9b-abliterated:256k' is listed on the shaper host
    but could not be loaded (HTTP 500); the box has no room for it right now

"The shaper host"/"the box" was unambiguous only while there was one. There are
now two ollama daemons in the fleet with directly opposed ability to serve this
model, measured the same hour:

    192.168.1.124:11434 (3090)  cannot load ANY model -- HTTP 500
                                `cudaMemGetInfo failed (out of memory)` while
                                Inkling-2-Mini holds the 23 GB card
    192.168.1.232:11434 (5090)  loads qwen2.5:7b in 2.28 s and shapes correctly
                                (202 prompt / 27 completion tokens, 1.25x)

SHAPE_OLLAMA selects between them, so a skip that omits the address leaves the
operator unable to tell "the default box is full" from "I pointed it at the wrong
box". The census that found this: of five ShaperUnavailable raise sites, two
named no host (shape.py:259 HTTP 404 at generation time, shape.py:269 load
failure) and only those two could fire at generation rather than preflight.

What is asserted
----------------
Both generation-time skips carry host:port, the address is the one actually
being talked to (honours SHAPE_OLLAMA, not a hardcoded default), and the 404 and
load-failure paths stay distinguishable from each other.
"""
import sys
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading

from _pytest.monkeypatch import MonkeyPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "comms-lane"))
import shape  # noqa: E402


class _Refuse(BaseHTTPRequestHandler):
    """A shaper host that lists the model but refuses to generate it."""

    status = 500
    body = b'{"error":"failed to allocate CUDA0 buffer"}'

    def do_GET(self):                       # noqa: N802  (/api/tags)
        doc = b'{"models":[{"name":"qwen2.5:7b"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(doc)))
        self.end_headers()
        self.wfile.write(doc)

    def do_POST(self):                      # noqa: N802  (/api/generate)
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *a):
        pass


class _Host:
    def __init__(self, status=500, body=b'{"error":"failed to allocate CUDA0 buffer"}'):
        cls = type("_S", (_Refuse,), {"status": status, "body": body})
        self.server = HTTPServer(("127.0.0.1", 0), cls)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class SkipNamesTheHostTests(unittest.TestCase):
    def setUp(self):
        self.mp = MonkeyPatch()
        self.mp.setenv("SHAPE_PROBE_TIMEOUT", "5")

    def tearDown(self):
        self.mp.undo()

    def test_a_load_failure_names_the_host_it_tried(self):
        with _Host() as h:
            self.mp.setattr(shape, "OLLAMA", f"http://127.0.0.1:{h.port}")
            self.mp.setattr(shape, "MODEL", "qwen2.5:7b")
            with self.assertRaises(shape.ShaperUnavailable) as ctx:
                shape.shape("say ok")
            msg = str(ctx.exception)
            self.assertIn(f"127.0.0.1:{h.port}", msg,
                          "the skip must name the address that refused it, not "
                          f"'the shaper host'; got: {msg}")
            self.assertIn("could not be loaded", msg)

    def test_the_named_host_tracks_the_configured_one(self):
        """Two different hosts must produce two different messages."""
        seen = set()
        for _ in range(2):
            with _Host() as h:
                self.mp.setattr(shape, "OLLAMA", f"http://127.0.0.1:{h.port}")
                self.mp.setattr(shape, "MODEL", "qwen2.5:7b")
                with self.assertRaises(shape.ShaperUnavailable) as ctx:
                    shape.shape("say ok")
                seen.add(str(ctx.exception))
        self.assertEqual(len(seen), 2,
                         "a constant message cannot tell two hosts apart: "
                         f"{seen}")

    def test_a_404_names_the_host_and_stays_distinct(self):
        with _Host(status=404) as h:
            self.mp.setattr(shape, "OLLAMA", f"http://127.0.0.1:{h.port}")
            self.mp.setattr(shape, "MODEL", "qwen2.5:7b")
            with self.assertRaises(shape.ShaperUnavailable) as ctx:
                shape.shape("say ok")
            msg = str(ctx.exception)
            self.assertIn(f"127.0.0.1:{h.port}", msg)
            self.assertIn("404", msg)
            self.assertNotIn("could not be loaded", msg,
                             "a missing model must not be reported as a full box")

    def test_an_unrelated_500_still_raises(self):
        """The narrowness this lane insists on, re-asserted at the new message."""
        with _Host(status=500, body=b'{"error":"something else broke"}') as h:
            self.mp.setattr(shape, "OLLAMA", f"http://127.0.0.1:{h.port}")
            self.mp.setattr(shape, "MODEL", "qwen2.5:7b")
            with self.assertRaises(Exception) as ctx:
                shape.shape("say ok")
            self.assertNotIsInstance(
                ctx.exception, shape.ShaperUnavailable,
                "an unrelated server bug must not become silent capacity loss")


if __name__ == "__main__":
    unittest.main()
