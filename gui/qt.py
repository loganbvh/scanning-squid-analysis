from qtpy import QtGui, QtCore, QtWidgets, API
from qtpy.QtCore import Qt

if API == 'pyqt5':
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
else:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT

__all__ = ['QtGui', 'QtCore', 'QtWidgets', 'Qt',
           'run_on_qthread', 'FigureCanvasQTAgg', 'NavigationToolbar2QT']
           
class QThreadWorker(QtCore.QObject):
    finished = QtCore.Signal()
    def __init__(self, func, args):
        super(QThreadWorker, self).__init__()
        self.func = func
        self.args = args
        self.result = None

    def do_work(self):
        self.result = self.func(*self.args)
        self.finished.emit()

_WORKERS = []
_THREADS = []

def run_on_qthread(func, args, on_finished=None):
    worker = QThreadWorker(func, args)
    thread = QtCore.QThread()
    worker.moveToThread(thread)
    _WORKERS.append(worker)
    _THREADS.append(thread)

    def finished():
        _WORKERS.remove(worker)
        _THREADS.remove(thread)
        if on_finished is not None:
            on_finished(worker.result)
        thread.quit()

    thread.started.connect(worker.do_work)
    worker.finished.connect(finished)
    thread.start()