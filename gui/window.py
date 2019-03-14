import os
import sys
from qt import QtGui, QtWidgets, QtCore, Qt
import matplotlib.pyplot as plt
from widgets import (QIPythonWidget, MetaWidget, DataSetBrowser, DataSetPlotter)
from utils import load_json_ordered

app = QtWidgets.QApplication([])

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.setWindowTitle('scanning-squid data viewer')
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        app.setWindowIcon(QtGui.QIcon(icon_path))

        self.shell = QIPythonWidget()
        self.station_snap = MetaWidget()
        self.measurement_meta = MetaWidget()
        self.dataset_browser = DataSetBrowser()
        self.dataset_browser.dataset_selector.clicked.connect(self.load_dataset)
        self.dataset_browser.dataset_selector.doubleClicked.connect(self.update_dataset_plot)
        self.dataset_plotter = DataSetPlotter()

        import numpy
        self.shell.pushVariables({
            'np': numpy,
            'plt': plt
            })

        self.file_menu = self.menuBar().addMenu('File')
        self.view_menu = self.menuBar().addMenu('View')
        self.plot_menu = self.menuBar().addMenu('Plot')

        self.dataset_dock = self.add_dock(self.dataset_browser, 'DataSet Browser', 'Left')
        self.snapshot_dock = self.add_dock(self.station_snap, 'Microscope Snapshot', 'Left')
        self.meta_dock = self.add_dock(self.measurement_meta, 'Measurement Metadata', 'Left')
        self.shell_dock = self.add_dock(self.shell, 'Shell', 'Left')
        self.plotter_dock = self.add_dock(self.dataset_plotter, 'DataSetPlotter', 'Right')
        self.tabifyDockWidget(self.snapshot_dock, self.meta_dock)

        for _, action in self.dataset_plotter.opt_checks.items():
            self.plot_menu.addAction(action)

        self.plot_menu.addAction('Export matplotlib...', self.dataset_plotter.toolbar.save_figure,
                                    QtGui.QKeySequence('Ctrl+S'))
        self.plot_menu.addAction('Export pyqtgraph...', self.dataset_plotter.export_pg,
                                    QtGui.QKeySequence('Ctrl+Shift+S'))                
        self.file_menu.addAction('Select directory...', self.dataset_browser.select_from_dialog,
                                    QtGui.QKeySequence('Ctrl+O'))
        

    def load_dataset(self):
        dataset = self.dataset_browser.get_dataset()
        if 'snapshot.json' not in os.listdir(dataset.location):
            return
        path = os.path.join(dataset.location, 'snapshot.json')
        meta = load_json_ordered(path)
        snap = meta.pop('station')
        self.station_snap.load_meta(data=snap)
        self.measurement_meta.load_meta(data=meta)
        self.shell.pushVariables({'dataset': dataset})
        self.dataset = dataset
        self.shell.pushVariables({'dataset': self.dataset})

    def update_dataset_plot(self):
        self.load_dataset()
        if self.dataset is None or 'snapshot.json' not in os.listdir(self.dataset.location):
            return
        self.dataset_plotter.update(self.dataset)
        self.shell.pushVariables({'arrays': self.dataset_plotter.arrays})

    def add_dock(self, widget, name, location, min_width=None):
        dock = QtWidgets.QDockWidget(name)
        dock.setWidget(widget)
        self.view_menu.addAction(dock.toggleViewAction())
        loc_const = getattr(Qt, location+'DockWidgetArea')
        self.addDockWidget(loc_const, dock)
        if min_width is not None:
            dock.setMinimumWidth(min_width)
        return dock

def main():
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.showMaximized()

    def test():
        pass

    app.lastWindowClosed.connect(sys.exit)
    QtCore.QTimer.singleShot(100, test)
    app.exec_()
    
if __name__ == '__main__':
    main()