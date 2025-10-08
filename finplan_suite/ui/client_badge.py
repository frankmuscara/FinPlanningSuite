# finplan_suite/ui/client_badge.py
from PyQt6.QtWidgets import QLabel
from .event_bus import bus, get_current_client

def format_client(c) -> str:
    if not c:
        return "No client selected"
    name = f"{(c.last_name or '').strip()}, {(c.first_name or '').strip()}".strip(", ")
    short = c.client_id[:8] if getattr(c, "client_id", None) else "—"
    return f"Client: {name}  [{short}]"

class ClientBadge(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("color:#666; font-style: italic;")
        self.setText(format_client(get_current_client()))
        bus.clientChanged.connect(self._on_client_changed)

    def _on_client_changed(self, c):
        self.setText(format_client(c))
