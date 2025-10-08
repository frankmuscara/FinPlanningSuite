# finplan_suite/ui/event_bus.py
from PyQt6.QtCore import QObject, pyqtSignal

class EventBus(QObject):
    clientChanged = pyqtSignal(object)   # emits a Client or None

bus = EventBus()

# simple in-memory holder so late subscribers can read current client
_current_client = None

def set_current_client(c):
    global _current_client
    _current_client = c
    bus.clientChanged.emit(c)

def get_current_client():
    return _current_client
