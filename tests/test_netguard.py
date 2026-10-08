"""The network guard must block every non-loopback connection and allow loopback."""

import socket
import threading

import pytest

from ratio import netguard


@pytest.fixture(autouse=True)
def guard_installed():
    netguard.install()
    yield
    netguard.install()  # the suite-wide guard stays on for later tests


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "[::1]", "127.0.0.5"])
def test_loopback_hosts_are_allowed(host):
    assert netguard.is_loopback_host(host)


@pytest.mark.parametrize("host", ["8.8.8.8", "93.184.216.34", "example.com", "huggingface.co", "", None])
def test_other_hosts_are_not_loopback(host):
    assert not netguard.is_loopback_host(host)


def test_outbound_tcp_connect_is_blocked():
    with pytest.raises(netguard.NetworkBlockedError):
        socket.create_connection(("93.184.216.34", 80), timeout=1)


def test_dns_lookup_of_external_host_is_blocked():
    with pytest.raises(netguard.NetworkBlockedError):
        socket.getaddrinfo("huggingface.co", 443)


def test_udp_connect_to_external_host_is_blocked():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(netguard.NetworkBlockedError):
            sock.connect(("8.8.8.8", 1))
    finally:
        sock.close()


def test_loopback_connection_still_works():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted = threading.Thread(target=lambda: server.accept()[0].close())
    accepted.start()
    client = socket.create_connection(("127.0.0.1", port), timeout=2)
    client.close()
    accepted.join(timeout=2)
    server.close()


def test_install_is_idempotent_and_uninstall_restores():
    netguard.install()
    netguard.install()
    assert netguard.is_installed()
    netguard.uninstall()
    assert not netguard.is_installed()
    assert socket.getaddrinfo is not None
    netguard.install()
    assert netguard.is_installed()


def test_udp_sendto_external_host_is_blocked():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(netguard.NetworkBlockedError):
            sock.sendto(b"x", ("8.8.8.8", 53))
    finally:
        sock.close()


def test_gethostbyname_of_external_host_is_blocked():
    with pytest.raises(netguard.NetworkBlockedError):
        socket.gethostbyname("example.com")
