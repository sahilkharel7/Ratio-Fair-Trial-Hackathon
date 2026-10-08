"""Block every outbound network connection except loopback.

Hard rule: no network calls besides the local Ollama server. The app, the eval and the test
suite install this guard at startup, so an accidental call fails loudly instead of leaking
monitoring data. Unix-domain sockets and binding local servers are unaffected.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})
_originals: dict[str, Any] = {}


class NetworkBlockedError(ConnectionRefusedError):
    """Raised when code tries to reach a host other than this machine."""


def is_loopback_host(host: object) -> bool:
    if host is None:
        return False
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    name = str(host).strip().strip("[]").lower()
    if not name:
        return False
    if name in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(name.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _blocked(host: object) -> NetworkBlockedError:
    return NetworkBlockedError(
        f"Ratio blocks outbound network access (attempted {host!r}); only loopback is allowed."
    )


def _check_address(address: object) -> None:
    if isinstance(address, (str, bytes)):  # AF_UNIX path
        return
    if isinstance(address, tuple) and address and not is_loopback_host(address[0]):
        raise _blocked(address[0])


def install() -> None:
    """Patch socket connect and DNS lookup. Safe to call more than once."""
    if _originals:
        return
    _originals.update(
        connect=socket.socket.connect,
        connect_ex=socket.socket.connect_ex,
        sendto=socket.socket.sendto,
        sendmsg=socket.socket.sendmsg,
        getaddrinfo=socket.getaddrinfo,
        gethostbyname=socket.gethostbyname,
        gethostbyname_ex=socket.gethostbyname_ex,
    )

    def guarded_connect(self: socket.socket, address: Any) -> None:
        _check_address(address)
        return _originals["connect"](self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        _check_address(address)
        return _originals["connect_ex"](self, address)

    def guarded_sendto(self: socket.socket, data: Any, *args: Any) -> int:
        _check_address(args[-1])  # the address is always the last argument
        return _originals["sendto"](self, data, *args)

    def guarded_sendmsg(self: socket.socket, buffers: Any, *args: Any) -> int:
        if len(args) >= 3 and args[2] is not None:
            _check_address(args[2])
        return _originals["sendmsg"](self, buffers, *args)

    def guarded_lookup(name: str) -> Any:
        def lookup(host: Any, *args: Any, **kwargs: Any) -> Any:
            if host is not None and not is_loopback_host(host):
                raise _blocked(host)
            return _originals[name](host, *args, **kwargs)

        return lookup

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.socket.sendto = guarded_sendto  # type: ignore[method-assign]
    socket.socket.sendmsg = guarded_sendmsg  # type: ignore[method-assign]
    socket.getaddrinfo = guarded_lookup("getaddrinfo")
    socket.gethostbyname = guarded_lookup("gethostbyname")
    socket.gethostbyname_ex = guarded_lookup("gethostbyname_ex")


def uninstall() -> None:
    if not _originals:
        return
    socket.socket.connect = _originals["connect"]  # type: ignore[method-assign]
    socket.socket.connect_ex = _originals["connect_ex"]  # type: ignore[method-assign]
    socket.socket.sendto = _originals["sendto"]  # type: ignore[method-assign]
    socket.socket.sendmsg = _originals["sendmsg"]  # type: ignore[method-assign]
    socket.getaddrinfo = _originals["getaddrinfo"]
    socket.gethostbyname = _originals["gethostbyname"]
    socket.gethostbyname_ex = _originals["gethostbyname_ex"]
    _originals.clear()


def is_installed() -> bool:
    return bool(_originals)
