"""Native process accounting and parent lifetime for isolated video encoders."""

import ctypes
import errno
import os
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path


class _TaskInfo(ctypes.Structure):
    # Darwin proc_taskinfo ABI (bsd/sys/proc_info.h), identical on arm64 and x86_64.
    _fields_ = [
        ("virtual_bytes", ctypes.c_uint64),
        ("resident_bytes", ctypes.c_uint64),
        ("_times", ctypes.c_uint64 * 4),
        ("_counters", ctypes.c_int32 * 12),
    ]


@lru_cache(maxsize=1)
def _libproc():
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_pidinfo.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
    ]
    library.proc_pidinfo.restype = ctypes.c_int
    return library


class _ProcessMemory(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("page_faults", ctypes.c_uint32),
        ("peak_working_set", ctypes.c_size_t),
        ("working_set", ctypes.c_size_t),
        ("_other_counters", ctypes.c_size_t * 6),
    ]


@lru_cache(maxsize=1)
def _kernel():
    from ctypes import wintypes

    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    library.OpenProcess.restype = wintypes.HANDLE
    library.CloseHandle.argtypes = [wintypes.HANDLE]
    library.PeekNamedPipe.argtypes = [
        wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
        wintypes.LPDWORD, wintypes.LPDWORD, wintypes.LPDWORD,
    ]
    library.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_ProcessMemory), wintypes.DWORD,
    ]
    return library


def resident_bytes(pid):
    """Current resident bytes, rather than ru_maxrss's historical high-water mark."""
    if sys.platform == "darwin":
        info = _TaskInfo()
        size = ctypes.sizeof(info)
        read = _libproc().proc_pidinfo(pid, 4, 0, ctypes.byref(info), size)
        if read != size:
            error = ctypes.get_errno() if read <= 0 else errno.EPROTO
            raise OSError(error or errno.ESRCH, "Could not read encoder resident memory")
        return info.resident_bytes
    if sys.platform.startswith("linux"):
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        raise OSError(errno.ESRCH, "Encoder resident memory is unavailable")
    if sys.platform == "win32":
        library = _kernel()
        process = library.OpenProcess(0x1000, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            counters = _ProcessMemory(size=ctypes.sizeof(_ProcessMemory))
            if not library.K32GetProcessMemoryInfo(process, ctypes.byref(counters), counters.size):
                raise ctypes.WinError(ctypes.get_last_error())
            return counters.working_set
        finally:
            library.CloseHandle(process)
    raise OSError(errno.ENOSYS, "无法在此系统读取编码进程内存")


def watch_parent(fd, parent_pid):
    """录制进程是管道唯一写端，异常退出后编码器也能收到 EOF。"""
    if os.getppid() != parent_pid:
        raise RuntimeError("Recording parent exited before encoder startup")

    def watch():
        try:
            if sys.platform == "win32":
                import msvcrt

                handle = msvcrt.get_osfhandle(fd)
                peek = _kernel().PeekNamedPipe
                # 这条管道只用来检测父进程退出，避免 Windows CRT 阻塞在标准输入上。
                while peek(handle, None, 0, None, None, None):
                    time.sleep(0.1)
                return
            while os.read(fd, 1):
                pass
        finally:
            os.close(fd)
            # A daemon cannot wait for journal readers or a native codec to finish.
            os._exit(1)

    thread = threading.Thread(target=watch, name="AlohaMiniEncoderParent", daemon=True)
    thread.start()
    return thread
