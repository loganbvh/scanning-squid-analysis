import os
from ..qt import *
import qcodes as qc
from ..qjsonmodel import QJsonModel
from .plots import DataSetPlotter
from ..utils import load_json_ordered

__all__ = ['DataSetBrowser']

class DataSetBrowser(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.directory = None
        self.date_selector = QtWidgets.QComboBox(parent=self)
        self.dataset_selector = QtWidgets.QListView(parent=self)
        self.dataset_list = QtGui.QStandardItemModel(parent=self)
        self.dataset_selector.setModel(self.dataset_list)
        self.dataset_selector.setIconSize(QtCore.QSize(150, 150))

        button_widget = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout()
        button_widget.setLayout(button_layout)
        select_dir_button = QtWidgets.QPushButton('Select directory')
        select_dir_button.clicked.connect(self.select_from_dialog)
        refresh_button = QtWidgets.QPushButton('Refresh')
        button_layout.addWidget(select_dir_button)
        button_layout.addWidget(refresh_button)
        #widget = QtWidgets.QWidget(parent=self)
        selector_layout = QtWidgets.QVBoxLayout(self)
        selector_layout.addWidget(button_widget)
        # selector_layout.addWidget(select_dir_button)
        # selector_layout.addWidget(refresh_button)
        selector_layout.addWidget(self.date_selector)
        selector_layout.addWidget(self.dataset_selector)
        #self.addWidget(widget)

        self.date_selector.currentIndexChanged.connect(self.set_available_datasets)
        refresh_button.clicked.connect(self.set_available_dates)

    def select_from_dialog(self):
        self.directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select data directory')
        self.set_available_dates()

    def set_available_dates(self):
        self.date_selector.clear()
        if not os.path.isdir(self.directory):
            return
        for item in sorted(os.listdir(self.directory), reverse=True):
            if item.startswith('.'):
                continue
            path = os.path.join(self.directory, item)
            if os.path.isdir(path):
                self.date_selector.addItem(item)

    def get_dataset(self):
        self.dataset_name = self.dataset_selector.currentIndex().data()
        if not self.dataset_name or not self.date_name:
            return None
        fpath = os.path.join(self.directory, self.date_name, self.dataset_name)
        return qc.load_data(location=fpath)

    def set_available_datasets(self, index):
        self.dataset_list.clear()
        self.date_name = str(self.date_selector.currentText())
        if not self.date_name:
            return None
        fpath = os.path.join(self.directory, self.date_name)
        items = sorted([d for d in os.listdir(fpath) if os.path.isdir(os.path.join(fpath, d))], reverse=True)

        for d in items:
            item = QtGui.QStandardItem(d)
            item.setEditable(False)
            image_list = [f for f in os.listdir(os.path.join(fpath, d)) if f.endswith('.png')]
            if image_list:
                icon_path = os.path.join(fpath, d, image_list[0])
                item.setIcon(QtGui.QIcon(icon_path))
            self.dataset_list.appendRow(item)