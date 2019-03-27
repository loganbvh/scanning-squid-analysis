from qtpy import QtGui, QtCore, QtWidgets, API
from qtpy.QtCore import Qt

if API == 'pyqt5':
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
else:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT

__all__ = ['QtGui', 'QtCore', 'QtWidgets', 'Qt', 'FigureCanvasQTAgg', 'NavigationToolbar2QT']