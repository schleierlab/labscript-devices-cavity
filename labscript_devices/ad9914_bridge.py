r"""64-bit client half of the AD9914 32-bit DLL bridge.

Spawns ad9914_bridge32.py under a 32-bit interpreter and forwards the eleven
adiddseval.dll entry points to it. See ad9914_bridge32.py for why the split is
at this particular level.

Transport: this process binds a listening socket on 127.0.0.1 with an ephemeral
port, passes the port to the child on its command line, and the child connects
back. Nothing depends on the child's stdout, which matters because the ADI DLL
prints to the console. Messages are newline-delimited JSON; every argument and
return value is a plain int or bool.

Configure the interpreter with the AD9914_PYTHON32 environment variable, or via
the connection table (see AD9914Worker), or leave it and the default below is
used.
"""

import json
import os
import socket
import subprocess
import sys
import threading

DEFAULT_PYTHON32 = r"C:\labscript-suite\tools\python-3.11.9-embed-win32\python.exe"
BRIDGE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "ad9914_bridge32.py")

STARTUP_TIMEOUT = 30.0
CALL_TIMEOUT = 60.0


class BridgeError(RuntimeError):
    """Raised in this process for a failure that happened in the 32-bit half.

    The remote traceback is attached as .remote_traceback so it shows up in the
    BLACS tab rather than being swallowed.
    """

    def __init__(self, message, remote_traceback=""):
        super().__init__(message)
        self.remote_traceback = remote_traceback


class AD9914Bridge(object):
    """Proxy for one board, backed by one 32-bit child process.

    One bridge per BLACS worker, matching how the two DDS tabs already run as
    two independent processes. Board handles are process-local, so they are
    never shared; the only cross-process coordination is a named mutex the
    child takes across enumeration.
    """

    def __init__(self, instance, python32=None, dll_dir=None):
        self.instance = instance
        self.python32 = python32 or os.environ.get(
            "AD9914_PYTHON32", DEFAULT_PYTHON32
        )
        self._next_id = 0
        self._lock = threading.Lock()
        self._proc = None
        self._conn = None
        self._listener = None
        self.child_pid = None
        self._start()

    # -- lifecycle ------------------------------------------------------

    def _start(self):
        if not os.path.isfile(self.python32):
            raise BridgeError(
                "32-bit Python not found at %r. Install the win32 embeddable "
                "build there, or set AD9914_PYTHON32." % self.python32
            )
        if not os.path.isfile(BRIDGE_SCRIPT):
            raise BridgeError("bridge script missing: %r" % BRIDGE_SCRIPT)

        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self._listener.settimeout(STARTUP_TIMEOUT)
        port = self._listener.getsockname()[1]

        self._proc = subprocess.Popen(
            [self.python32, BRIDGE_SCRIPT,
             "--port", str(port), "--instance", str(self.instance)],
            cwd=os.path.dirname(BRIDGE_SCRIPT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            sock, _ = self._listener.accept()
        except socket.timeout:
            stderr = self._drain_stderr()
            self._proc.kill()
            raise BridgeError(
                "32-bit bridge did not connect back within %.0f s.%s"
                % (STARTUP_TIMEOUT, stderr)
            )
        finally:
            self._listener.close()
            self._listener = None

        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(CALL_TIMEOUT)
        self._conn = sock.makefile("rwb")

        hello = self._read_message()
        if hello.get("event") == "fatal":
            raise BridgeError(
                "32-bit bridge failed to load the DLL",
                hello.get("error", ""),
            )
        if hello.get("event") != "ready":
            raise BridgeError("unexpected greeting from bridge: %r" % hello)
        self.child_pid = hello.get("pid")

    def _drain_stderr(self):
        if self._proc is None or self._proc.stderr is None:
            return ""
        try:
            data = self._proc.stderr.read()
        except Exception:
            return ""
        if not data:
            return ""
        return "\nChild stderr:\n" + data.decode("utf-8", "replace")

    def close(self):
        if self._conn is not None:
            try:
                self._call("shutdown")
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- transport ------------------------------------------------------

    def _read_message(self):
        line = self._conn.readline()
        if not line:
            stderr = self._drain_stderr()
            raise BridgeError("32-bit bridge closed the connection." + stderr)
        return json.loads(line.decode("utf-8"))

    def _call(self, method, *args):
        with self._lock:
            if self._conn is None:
                raise BridgeError("bridge is not running")
            self._next_id += 1
            req_id = self._next_id
            payload = {"id": req_id, "method": method, "args": list(args)}
            self._conn.write(json.dumps(payload).encode("utf-8") + b"\n")
            self._conn.flush()

            while True:
                msg = self._read_message()
                if msg.get("id") == req_id:
                    break

        if not msg.get("ok", False):
            raise BridgeError(msg.get("error", "unknown bridge error"),
                              msg.get("traceback", ""))
        return msg.get("result")

    # -- the eleven DLL entry points ------------------------------------

    def find_hardware(self, vid, pid, instance):
        return self._call("find_hardware", vid, pid, instance)

    def is_connected(self):
        return self._call("is_connected")

    def get_port_config(self, port):
        return self._call("get_port_config", port)

    def set_port_config(self, port, value):
        return self._call("set_port_config", port, value)

    def get_port_value(self, port):
        return self._call("get_port_value", port)

    def set_port_value(self, port, data):
        return self._call("set_port_value", port, data)

    def get_spi_instruction(self, rw, addr):
        return self._call("get_spi_instruction", rw, addr)

    def spi_read(self, instr, reg_length, flag=0):
        return self._call("spi_read", instr, reg_length, flag)

    def spi_write(self, data_u64, length, flag=0):
        return self._call("spi_write", data_u64, length, flag)

    def ping(self):
        return self._call("ping")


if __name__ == "__main__":
    # Smoke test: python ad9914_bridge.py [instance]
    inst = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    b = AD9914Bridge(inst)
    print("bridge up:", b.ping())
    print("boards found:", b.find_hardware(0x0456, 0xEE1F, inst))
    print("is_connected:", b.is_connected())
    b.close()
