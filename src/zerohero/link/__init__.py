"""Lead/follow link over TCP (WiFi/LAN/Tailscale) or Bluetooth RFCOMM. See ARCHITECTURE.md "link"."""

from zerohero.link.discovery import discover
from zerohero.link.follow import Follower
from zerohero.link.lead import Lead
from zerohero.link.transport import BluetoothUnavailable, parse_target

__all__ = ["BluetoothUnavailable", "Follower", "Lead", "discover", "parse_target"]
