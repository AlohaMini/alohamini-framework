"""录制、训练和环境同步共用的文件锁及落盘操作。"""

import errno
import os

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("internal", ctypes.c_size_t),
            ("internal_high", ctypes.c_size_t),
            ("offset", wintypes.DWORD),
            ("offset_high", wintypes.DWORD),
            ("event", wintypes.HANDLE),
        ]

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel.LockFileEx.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_Overlapped),
    ]
    _kernel.UnlockFileEx.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, ctypes.POINTER(_Overlapped),
    ]

if os.name != "nt":
    import fcntl


def lock_file(stream, *, exclusive=True, blocking=False):
    """锁定已打开的文件，关闭文件时释放。"""
    if os.name != "nt":
        flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(stream, flags if blocking else flags | fcntl.LOCK_NB)
        return

    handle = msvcrt.get_osfhandle(stream.fileno())
    flags = (0 if blocking else 0x01) | (0x02 if exclusive else 0)
    if _kernel.LockFileEx(handle, flags, 0, 1, 0, ctypes.byref(_Overlapped())):
        return
    error = ctypes.get_last_error()
    if error == 33:
        raise BlockingIOError(errno.EAGAIN, "文件锁正被使用")
    raise ctypes.WinError(error)


def unlock_file(stream):
    if os.name == "nt":
        handle = msvcrt.get_osfhandle(stream.fileno())
        if not _kernel.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(_Overlapped())):
            raise ctypes.WinError(ctypes.get_last_error())
        return
    fcntl.flock(stream, fcntl.LOCK_UN)


def sync_directory(path):
    if os.name == "nt":
        # Windows 的普通文件描述符不能打开目录，文件内容继续由写入方刷盘。
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
