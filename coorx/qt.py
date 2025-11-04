import contextlib
import importlib
import sys

HAVE_QT = None
QtGui = None
qt_lib_order = ['PyQt6', 'PySide6', 'PyQt5', 'PySide2']


def check_qt_imported():
    global HAVE_QT, QtGui
    if HAVE_QT is None:
        for qt_lib in qt_lib_order:
            if f'{qt_lib}.QtGui' in sys.modules:
                HAVE_QT = True
                QtGui = importlib.import_module(f'{qt_lib}.QtGui')
                break
    return HAVE_QT


check_qt_imported()


def import_qt_gui():
    """Import QtGui and return it."""
    global HAVE_QT, QtGui
    if not HAVE_QT:
        check_qt_imported()

    if QtGui is not None:
        return QtGui

    if not HAVE_QT:
        HAVE_QT = False
        for qt_lib in qt_lib_order:
            with contextlib.suppress(ImportError):
                QtGui = importlib.import_module(f'{qt_lib}.QtGui')
                HAVE_QT = True
                return QtGui

    raise ImportError(f'No importable Qt library found (tried {", ".join(qt_lib_order)})')
