r"""32-bit bridge server for the AD9914 eval board.

adiddseval.dll and its dependency ADI_CYUSB_USB4.dll are PE32 (i386) only, so
they cannot be loaded from the 64-bit interpreter that runs the rest of the
labscript suite. This module runs under a 32-bit Python and exposes the eleven
DLL entry points that AD9914Worker uses over a loopback socket.

It deliberately imports ONLY the standard library -- no numpy, no h5py, no
labscript. That is the whole point of bridging at this level: h5py has shipped
no 32-bit Windows wheel since 2.10.0/cp38 (2019), so anything that needs h5py
cannot live on this side. Everything above the DLL (register packing, FTW/POW/
ASF arithmetic, the HDF5 shot file) stays in the 64-bit worker.

Started by ad9914_bridge.py in the 64-bit process; not meant to be run by hand
except for debugging:

    python-3.11.9-embed-win32\python.exe ad9914_bridge32.py --port 5xxxx --instance 0

The ctypes calls below are copied verbatim from AD9914Worker. In particular no
argtypes/restype are declared: every argument is an int or a byref pointer, so
the ctypes defaults are already correct at 32-bit. Do not "fix" this.
"""

import json
import os
import socket
import struct
import sys
import traceback
from ctypes import (
    windll, byref, sizeof, c_int, c_byte, c_uint32, c_uint64, c_void_p,
    c_wchar_p, c_bool,
)

DLL_DIR = os.path.dirname(os.path.abspath(__file__))
DLL_NAME = "adiddseval.dll"

# Serialises the enumeration phase across bridge processes. FindHardware calls
# Search_For_Boards, which walks every attached board; two processes doing that
# concurrently is the one real contention point. Steady-state register traffic
# is unaffected -- each process holds its own CreateFile handle to its own
# board and never touches the mutex again.
ENUM_MUTEX_NAME = "Global\\labscript_ad9914_enumeration"
ENUM_MUTEX_TIMEOUT_MS = 30000


class EnumerationLock(object):
    """Named-mutex context manager, so only one process enumerates at a time."""

    def __init__(self, name=ENUM_MUTEX_NAME, timeout_ms=ENUM_MUTEX_TIMEOUT_MS):
        self.name = name
        self.timeout_ms = timeout_ms
        self.handle = None
        self._k32 = windll.kernel32

    def __enter__(self):
        self._k32.CreateMutexW.restype = c_void_p
        self._k32.CreateMutexW.argtypes = [c_void_p, c_bool, c_wchar_p]
        self.handle = self._k32.CreateMutexW(None, False, self.name)
        if not self.handle:
            # Could not create the mutex (e.g. denied on Global\). Proceed
            # unserialised rather than failing outright.
            return self
        self._k32.WaitForSingleObject.argtypes = [c_void_p, c_int]
        result = self._k32.WaitForSingleObject(self.handle, self.timeout_ms)
        # 0 = acquired, 0x80 = abandoned by a crashed owner (still ours now).
        if result not in (0, 0x80):
            raise RuntimeError(
                "timed out waiting for AD9914 enumeration lock (%r)" % result
            )
        return self

    def __exit__(self, *exc):
        if self.handle:
            self._k32.ReleaseMutex.argtypes = [c_void_p]
            self._k32.ReleaseMutex(self.handle)
            self._k32.CloseHandle.argtypes = [c_void_p]
            self._k32.CloseHandle(self.handle)
            self.handle = None
        return False


class DDSBridge(object):
    """Owns the DLL and the board handle for one instance.

    The handle is a CreateFile handle and is only valid inside this process, so
    it never crosses the socket. Callers refer to "the board this bridge owns"
    implicitly.
    """

    def __init__(self):
        # Since Python 3.8 the directory of a DLL loaded by full path is not
        # searched for that DLL's own dependencies. ADI_CYUSB_USB4.dll sits
        # next to adiddseval.dll, so add the directory explicitly. (The
        # original worker got away with relying on the current directory.)
        os.add_dll_directory(DLL_DIR)
        self._dll = windll.LoadLibrary(os.path.join(DLL_DIR, DLL_NAME))

        self._fFindHardware = self._dll.FindHardware
        self._fGetHardwareHandles = self._dll.GetHardwareHandles
        self._fGetHardwareCount = self._dll.GetHardwareCount
        self._fIsConnected = self._dll.IsConnected
        self._fGetPortConfig = self._dll.GetPortConfig
        self._fSetPortConfig = self._dll.SetPortConfig
        self._fGetPortValue = self._dll.GetPortValue
        self._fSetPortValue = self._dll.SetPortValue
        self._fGetSpiInstruction = self._dll.GetSpiInstruction
        self._fSpiRead = self._dll.SpiRead
        self._fSpiWrite = self._dll.SpiWrite

        self.handle = None

    # -- RPC methods ----------------------------------------------------
    # Every argument and return value here is a plain int or bool, because all
    # the bytes-level packing stays in the 64-bit worker. Nothing needs
    # encoding beyond what JSON already does.

    def find_hardware(self, vid, pid, instance):
        vidArry = c_int * 1
        pidArry = c_int * 1
        vid_a = vidArry(vid)
        pid_a = pidArry(pid)
        length = c_int(1)

        with EnumerationLock():
            if not self._fFindHardware(byref(vid_a), byref(pid_a), length):
                raise RuntimeError("Could not find hardware")

            count = self._fGetHardwareCount()
            handleArray = c_int * count
            handles = handleArray(0)
            self._fGetHardwareHandles(byref(handles))

            if instance >= count:
                raise RuntimeError(
                    "Requested instance %d but only %d board(s) found"
                    % (instance, count)
                )
            self.handle = c_int(handles[instance])

        return count

    def is_connected(self):
        return bool(self._fIsConnected(self.handle))

    def get_port_config(self, port):
        val = c_byte()
        self._fGetPortConfig(self.handle, port, byref(val))
        return val.value

    def set_port_config(self, port, value):
        self._fSetPortConfig(self.handle, port, value)

    def get_port_value(self, port):
        data = c_byte()
        self._fGetPortValue(self.handle, port, byref(data))
        return data.value

    def set_port_value(self, port, data):
        self._fSetPortValue(self.handle, port, data)

    def get_spi_instruction(self, rw, addr):
        instr = c_byte()
        self._fGetSpiInstruction(rw, addr, byref(instr), sizeof(instr))
        return instr.value

    def spi_read(self, instr, reg_length, flag=0):
        """Returns the raw uint32 the DLL wrote into regVals.

        The worker turns this back into a bytearray with struct.pack('@I').
        """
        instr_c = c_byte(instr)
        regvals = c_uint32()
        self._fSpiRead(
            self.handle, byref(instr_c), sizeof(instr_c),
            byref(regvals), reg_length, flag,
        )
        return regvals.value

    def spi_write(self, data_u64, length, flag=0):
        """data_u64 is the already-packed 8-byte word, as an int.

        The worker builds it with struct.unpack('Q', ...) over the 4 data
        bytes plus the instruction byte plus 3 pad bytes, and passes
        length=regLength+1. Both are preserved exactly.
        """
        write_data = c_uint64(data_u64)
        self._fSpiWrite(self.handle, byref(write_data), length, flag)

    def ping(self):
        return {"pid": os.getpid(), "bits": struct.calcsize("P") * 8}


DISPATCH = (
    "find_hardware", "is_connected", "get_port_config", "set_port_config",
    "get_port_value", "set_port_value", "get_spi_instruction",
    "spi_read", "spi_write", "ping",
)


def serve(port, instance):
    sock = socket.create_connection(("127.0.0.1", port), timeout=30)
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn = sock.makefile("rwb")

    bridge = None
    try:
        bridge = DDSBridge()
        _send(conn, {"event": "ready", "pid": os.getpid(), "instance": instance})
    except Exception:
        _send(conn, {"event": "fatal", "error": traceback.format_exc()})
        return 1

    for line in conn:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line.decode("utf-8"))
        except Exception:
            continue

        req_id = req.get("id")
        method = req.get("method")
        args = req.get("args", [])

        if method == "shutdown":
            _send(conn, {"id": req_id, "ok": True, "result": None})
            break

        if method not in DISPATCH:
            _send(conn, {
                "id": req_id, "ok": False,
                "error": "unknown method %r" % method, "traceback": "",
            })
            continue

        try:
            result = getattr(bridge, method)(*args)
            _send(conn, {"id": req_id, "ok": True, "result": result})
        except Exception as exc:
            _send(conn, {
                "id": req_id, "ok": False,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            })
    return 0


def _send(conn, obj):
    conn.write(json.dumps(obj).encode("utf-8") + b"\n")
    conn.flush()


def main():
    if struct.calcsize("P") * 8 != 32:
        sys.exit("ad9914_bridge32 must run under 32-bit Python")
    port = int(sys.argv[sys.argv.index("--port") + 1])
    instance = int(sys.argv[sys.argv.index("--instance") + 1])
    sys.exit(serve(port, instance))


if __name__ == "__main__":
    main()
