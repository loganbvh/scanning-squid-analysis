import matplotlib.figure
from matplotlib import cm
from matplotlib import transforms
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 14})
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.linalg import lstsq
from scipy.ndimage.interpolation import rotate
from scipy import io
import h5py
import pyqtgraph.exporters
import pyqtgraph as pg
from ..qt import *
from .sliders import SliderWidget
from ..utils import *
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

mpl_cmaps = ('viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys')
pg_cmaps = ('thermal', 'flame', 'yellowy', 'bipolar', 'grey')#, 'spectrum', 'cyclic', 'greyclip')

__all__ = ['DataSetPlotter']

class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PlotWidget, self).__init__(parent=parent)
        self.current_data = None
        self.exp_data = {}
        self.fig = matplotlib.figure.Figure()
        self.fig.patch.set_alpha(1)
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.15)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.pyqt_plot = CrosshairPlotWidget(parent=self)
        self.pyqt_imview = CrossSectionImageView(parent=self)
        #self.raw_image_view = QtWidgets.QLabel()
        self.pyqt_plot.hide()
        self.pyqt_imview.hide()

        self.option_layout = QtWidgets.QHBoxLayout()

        self._setup_cmap()
        self._setup_background_subtraction()
        self._setup_transforms()
        self._setup_slices()
        self._setup_options()

        #self.raw_image_view.hide()
        self.slice_state = 1
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(self.option_layout)
        layout.addWidget(self.backsub_widget)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.pyqt_splitter = QtWidgets.QSplitter(Qt.Vertical, parent=self)
        self.pyqt_splitter.hide()
        layout.addWidget(self.pyqt_splitter)
        layout.addWidget(self.rotate_widget)
        
        pyqt_top_widgets = QtWidgets.QWidget(parent=self)
        pyqt_bottom_widgets = QtWidgets.QWidget(parent=self)
        pyqt_top_layout = QtWidgets.QVBoxLayout(pyqt_top_widgets)
        pyqt_bottom_layout = QtWidgets.QVBoxLayout(pyqt_bottom_widgets)
        pyqt_top_layout.addWidget(self.pyqt_plot)
        pyqt_top_layout.addWidget(self.pyqt_imview)
        pyqt_bottom_layout.addWidget(self.pyqt_imview.h_cross_section_widget)
        pyqt_bottom_layout.addWidget(self.pyqt_imview.v_cross_section_widget)
        self.pyqt_imview.h_cross_section_widget.hide()
        self.pyqt_imview.v_cross_section_widget.hide()
        self.pyqt_splitter.addWidget(pyqt_top_widgets)
        self.pyqt_splitter.addWidget(pyqt_bottom_widgets)
        self.pyqt_splitter.setStretchFactor(0, 1.5)
        self.pyqt_splitter.setStretchFactor(1,1)
        #layout.addWidget(self.raw_image_view)
        self.set_slice()

    def _setup_cmap(self):
        self.mpl_cmap = 'viridis'
        self.mpl_cmap_selector = QtWidgets.QComboBox()
        self.mpl_cmap_selector.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.pg_cmap_selector = QtWidgets.QComboBox()
        self.pg_cmap_selector.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        cmap_widget = QtWidgets.QGroupBox('Colormaps')
        # cmap_layout = QtWidgets.QGridLayout()
        # cmap_widget.setLayout(cmap_layout)
        # cmap_layout.addWidget(QtWidgets.QLabel('matplotlib'), 0, 0)
        # cmap_layout.addWidget(self.mpl_cmap_selector, 0, 1)
        # cmap_layout.addWidget(QtWidgets.QLabel('pyqtgraph'), 1, 0)
        # cmap_layout.addWidget(self.pg_cmap_selector, 1, 1)
        # cmap_widget.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        mpl_cmap_widget = QtWidgets.QGroupBox('matplotlib')
        mpl_cmap_layout = QtWidgets.QHBoxLayout(mpl_cmap_widget)
        mpl_cmap_layout.addWidget(self.mpl_cmap_selector)
        pg_cmap_widget = QtWidgets.QGroupBox('pyqtgraph')
        pg_cmap_layout = QtWidgets.QHBoxLayout(pg_cmap_widget)
        pg_cmap_layout.addWidget(self.pg_cmap_selector)
        cmap_layout = QtWidgets.QHBoxLayout(cmap_widget)
        mpl_cmap_widget.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        pg_cmap_widget.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        cmap_layout.addWidget(mpl_cmap_widget)
        cmap_layout.addWidget(pg_cmap_widget)
        self.option_layout.addWidget(cmap_widget)

        self.mpl_cmap_selector.currentIndexChanged.connect(self.set_cmap_mpl)
        for name in mpl_cmaps:
            self.mpl_cmap_selector.addItem(name)
        self.mpl_cmap_selector.setCurrentIndex(0)
        self.pg_cmap_selector.currentIndexChanged.connect(self.set_cmap_pg)
        for name in pg_cmaps:
            self.pg_cmap_selector.addItem(name)
        self.pg_cmap_selector.setCurrentIndex(0)

    def _setup_background_subtraction(self):
        self.backsub_radio = QtWidgets.QButtonGroup()
        backsub_buttons = [QtWidgets.QRadioButton(s) for s in ('none', 'min', 'max', 'mean', 'median', 'linear')]
        backsub_buttons[0].setChecked(True)
        self.backsub_widget = QtWidgets.QGroupBox('Background subtraction')
        self.backsub_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        backsub_layout = QtWidgets.QHBoxLayout(self.backsub_widget)
        for i, b in enumerate(backsub_buttons):
            backsub_layout.addWidget(b)
            self.backsub_radio.addButton(b, i)
        self.backsub_radio.buttonClicked.connect(self.replot)

        self.line_backsub_radio = QtWidgets.QButtonGroup()
        self.line_backsub_btn = QtWidgets.QCheckBox('line-by-line')
        self.line_backsub_btn.setChecked(False)
        self.x_line_backsub_btn, self.y_line_backsub_btn = xy_btns = [QtWidgets.QRadioButton(s) for s in ('x', 'y')]
        xy_btns[0].setChecked(True)
        backsub_layout.addWidget(self.line_backsub_btn)
        for i, b in enumerate(xy_btns):
            b.setDisabled(True)
            backsub_layout.addWidget(b)
            self.line_backsub_radio.addButton(b, i)
        
        self.line_backsub_btn.stateChanged.connect(self.update_line_by_line)
        self.line_backsub_radio.buttonClicked.connect(self.replot)

    def _setup_transforms(self):
        self.rotate_widget = QtWidgets.QGroupBox('Rotate')
        rotate_layout = QtWidgets.QVBoxLayout()
        self.rotate_widget.setLayout(rotate_layout)
        self.rotate_slider = SliderWidget(-180, 180, 0, 60)
        rotate_layout.addWidget(self.rotate_slider)
        self.rotate_slider.angle_box.valueChanged.connect(self.replot)

    def _setup_slices(self):
        self.slice_radio = QtWidgets.QButtonGroup()
        slice_buttons = [QtWidgets.QRadioButton(s) for s in ('none', 'x', 'y')]
        slice_buttons[0].setChecked(True)
        slice_widget = QtWidgets.QGroupBox('Slice')
        slice_layout = QtWidgets.QVBoxLayout(slice_widget)
        for i, b in enumerate(slice_buttons):
            slice_layout.addWidget(b)
            self.slice_radio.addButton(b, i+1)
        self.option_layout.addWidget(slice_widget)
        self.slice_radio.buttonClicked.connect(self.set_slice)
        slice_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)

    def _setup_options(self):
        opt_group = QtWidgets.QGroupBox('Plot Options')
        opt_layout = QtWidgets.QVBoxLayout()
        opt_group.setLayout(opt_layout)
        plot_opts = [
            ('pyqtgraph', False),
            ('histogram', True),
            ('zoom to fit', True),
            ('grid', True),
        ]
        self.opt_checks = {}
        for name, checked in plot_opts:
            btn = QtWidgets.QCheckBox(name)
            btn.setChecked(checked)
            opt_layout.addWidget(btn)
            btn.stateChanged.connect(self.replot)
            self.opt_checks[name] = btn
        self.option_layout.addWidget(opt_group)

    def set_cmap_mpl(self, idx):
        name = str(self.mpl_cmap_selector.itemText(idx))
        if not name:
            return
        self.mpl_cmap = name
        self.replot()

    def set_cmap_pg(self, idx):
        name = str(self.pg_cmap_selector.itemText(idx))
        if not name:
            return
        self.pyqt_imview.set_cmap(name)

    def get_opt(self, optname):
        return bool(self.opt_checks[optname].isChecked())

    def get_all_opts(self):
        return {name: self.get_opt(name) for name in self.opt_checks}

    def set_opts(self, opts):
        for name, val in opts.items():
            self.opt_checks[name].setChecked(val)

    def plot_arrays(self, xs, ys, zs=None, title=''):
        self.fig_title = title
        self.fig.clear()
        self.pyqt_plot.clear()
        self.toolbar.hide()
        self.canvas.hide()
        self.pyqt_splitter.hide()
        self.pyqt_plot.hide()
        self.pyqt_imview.hide()
        #self.raw_image_view.hide()

        # 1d data
        if zs is None:
            self.plot_1d(xs, ys)
        # 2d data
        else:
            angle = self.rotate_slider.angle_box.value()
            if self.slice_state == 1:
                slice_state = None
            elif self.slice_state == 2:
                slice_state = 'x'
            elif self.slice_state == 3:
                slice_state = 'y'
            self.plot_2d(xs, ys, zs, angle=angle, slice_state=slice_state)
        self.fig.suptitle(self.fig_title, fontsize=12)
        #self.fig.tight_layout()
        self.canvas.draw()

    def plot_1d(self, xs, ys, label='data'):
        label = ys[0]
        xlabel = f'{xs[0]} [{xs[1].units}]'
        ylabel = f'{ys[0]} [{ys[1].units}]'
        fmt = '.'
        ymin, ymax = np.min(ys[1]), np.max(ys[1])
        if self.get_opt('pyqtgraph'):
            self.pyqt_splitter.show()
            self.pyqt_plot.show()
            self.plot_1d_pg(xs, ys, xlabel, ylabel, label)
        else:
            self.toolbar.show()
            self.canvas.show()
            self.plot_1d_mpl(xs, ys, xlabel, ylabel, fmt, ymin, ymax, label)
        self.rotate_widget.hide()
        self.exp_data = {d[0]: {'array': d[1].magnitude, 'unit': str(d[1].units)} for d in (xs, ys)}

    def plot_1d_pg(self, xs, ys, xlabel, ylabel, label):
        self.pyqt_plot.setLabels(bottom=(xlabel,), left=(ylabel,))
        self.pyqt_plot.plot(xs[1], ys[1], symbol='o', pen=None)

    def plot_1d_mpl(self, xs, ys, xlabel, ylabel, fmt, ymin, ymax, label):
        ax = self.fig.add_subplot(111)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(xs[1], ys[1], fmt, label=label)
        if not self.get_opt('zoom to fit'):
            ax.set_ylim(ymin, ymax)
        ax.grid(self.get_opt('grid'))
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.9, bottom=0.15)
        ax.legend()

    def plot_2d(self, xs, ys, zs, cmap=None, angle=0, slice_state=None):
        cmap = cmap or self.mpl_cmap
        xlabel = f'{xs[0]} [{xs[1].units}]'
        ylabel = f'{ys[0]} [{ys[1].units}]'
        zlabel = f'{zs[0]} [{zs[1].units}]'       
        zmin, zmax = np.nanmin(zs[1]), np.nanmax(zs[1])
        self.rotate_widget.show()
        if self.get_opt('pyqtgraph'):
            self.pyqt_splitter.show()
            self.pyqt_imview.show()
            self.plot_2d_pg(xs, ys, zs, xlabel, ylabel, zlabel, angle=angle)
        else:
            self.toolbar.show()
            self.canvas.show()
            self.slider = self.plot_2d_mpl(xs, ys, zs, xlabel, ylabel, zlabel,
                                vmin=zmin, vmax=zmax, cmap=cmap, angle=angle, slice_state=slice_state)

    def plot_2d_pg(self, xs, ys, zs, xlabel, ylabel, zlabel, angle=0):
        pos = np.nanmin(xs[1][0].magnitude), np.nanmin(ys[1][0].magnitude)
        scale = np.ptp(xs[1].magnitude) / zs[1].shape[0], np.ptp(ys[1].magnitude) / zs[1].shape[1]
        z = rotate(zs[1].magnitude.T, angle, cval=np.nanmin(zs[1].magnitude))
        self.pyqt_imview.setImage(z, pos=pos, scale=scale)
        self.pyqt_imview.setLabels(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
        self.pyqt_imview.autoRange()
        self.pyqt_imview.set_histogram(self.get_opt('histogram'))
        self.exp_data = {d[0]: {'array': d[1].magnitude, 'unit': str(d[1].units)} for d in (xs, ys)}
        self.exp_data[zs[0]] = {'array': z, 'unit': str(zs[1].units)}

    def plot_2d_mpl(self, xs, ys, zs, xlabel, ylabel, zlabel, cmap=None, angle=0, slice_state=None, **kwargs):
        if slice_state is None:
            plt.rcParams.update({'font.size': 14})
            self.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.0, right=0.85, hspace=0.0, wspace=0)
            ax = self.fig.add_subplot(111)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            x0, y0 = np.mean(xs[1].magnitude), np.mean(ys[1].magnitude)
            tr = transforms.Affine2D().rotate_deg_around(x0, y0, angle)
            im = ax.pcolormesh(xs[1], ys[1], zs[1].magnitude, cmap=cmap,
                                transform=(tr + ax.transData), **kwargs)
            cbar = self.fig.colorbar(im)
            cbar.set_label(zlabel)
            ax.set_aspect('equal')
            z = rotate(zs[1].magnitude.T, angle, cval=np.nan)
            x = np.linspace(*ax.get_xlim(), z.shape[1])
            y = np.linspace(*ax.get_ylim(), z.shape[0])
            self.exp_data = {
                xs[0]: {'array': x, 'unit': str(xs[1].units)},
                ys[0]: {'array': y, 'unit': str(ys[1].units)},
                zs[0]: {'array': z, 'unit': str(zs[1].units)}
            }
        else:
            plt.rcParams.update({'font.size': 10})
            self.fig.subplots_adjust(top=0.9, bottom=0.05, left=0.0, right=1.0, hspace=0.5, wspace=0.0)
            ax0 = plt.subplot2grid((12,12), (0,2), colspan=6, rowspan=5, fig=self.fig)
            ax0.set_xlabel(xlabel)
            ax0.set_ylabel(ylabel)
            x0, y0 = np.mean(xs[1].magnitude), np.mean(ys[1].magnitude)
            tr = transforms.Affine2D().rotate_deg_around(x0, y0, angle)
            im = ax0.pcolormesh(xs[1], ys[1], zs[1].magnitude, cmap=cmap,
                                transform=(tr + ax0.transData), **kwargs)
            cbar = self.fig.colorbar(im)
            cbar.set_label(zlabel)
            ax0.set_aspect('equal')
            ax1 = plt.subplot2grid((12,12), (7,2), colspan=8, rowspan=5, fig=self.fig)
            ax1.grid(self.get_opt('grid'))
            xlab = xlabel if slice_state == 'x' else ylabel
            label = zlabel.split(' ')[:1] + [''.join(zlabel.split(' ')[1:])]
            ylab = '\n'.join(label)
            if slice_state == 'x':
                line, = ax1.plot(xs[1].magnitude, zs[1].magnitude[:,0], lw=3)
                cut = ax0.axhline(y=ax0.get_ylim()[0], color='k', alpha=0.8, lw=2)
            else:
                line, = ax1.plot(ys[1].magnitude, zs[1].magnitude[0,:], lw=3)
                cut = ax0.axvline(x=ax0.get_xlim()[0], color='k', alpha=0.8, lw=2)
            ax1.set_xlabel(xlab)
            ax1.set_ylabel(ylab)
            divider = make_axes_locatable(ax1)
            ax_slider = divider.append_axes('bottom', size='15%', pad=0.45)
            idx_label = 'y' if slice_state == 'x' else 'x'
            idx = 1 if slice_state == 'x' else 0
            slider = Slider(ax_slider, f'{idx_label} index', 0, zs[1].shape[idx],
                            valinit=0, valstep=1, valfmt='%i')
            z = rotate(zs[1].magnitude.T, angle, cval=np.nan)
            x = np.linspace(*ax0.get_xlim(), z.shape[1])
            y = np.linspace(*ax0.get_ylim(), z.shape[0])
            self.fig.tight_layout()
            def update(val):
                i = int(slider.val) - 1
                z = rotate(zs[1].magnitude.T, angle, cval=np.nan)
                x = np.linspace(*ax0.get_xlim(), z.shape[1])
                y = np.linspace(*ax0.get_ylim(), z.shape[0])
                if slice_state == 'x':
                    slider.valmax = len(y)
                    line.set_data(x, z[:,i])
                    ax1.set_xlim(np.nanmin(x), np.nanmax(x))
                    cut.set_ydata(2*[y[i]])
                else:
                    slider.valmax = len(x)
                    line.set_data(y, z[i,:])
                    ax1.set_xlim(np.nanmin(y), np.nanmax(y))
                    cut.set_xdata(2*[x[i]])
                slider.ax.set_xlim(slider.valmin,slider.valmax)
                ydata = line.get_ydata()
                vmin, vmax = np.nanmin(ydata), np.nanmax(ydata)
                margin = 0.1
                rng = vmax - vmin
                vmin = vmin - margin * rng
                vmax = vmax + margin * rng
                try:
                    ax1.set_ylim(vmin, vmax)
                except ValueError:
                    pass
                self.canvas.draw()
            slider.on_changed(update)
            self.exp_data = {
                xs[0]: {'array': x, 'unit': str(xs[1].units)},
                ys[0]: {'array': y, 'unit': str(ys[1].units)},
                zs[0]: {'array': z, 'unit': str(zs[1].units)}
            }
            if slice_state == 'x':
                self.exp_data['slice'] = {
                    xs[0]: {'array': line.get_xdata(), 'unit': str(xs[1].units)},
                    zs[0]: {'array': line.get_ydata(), 'unit': str(zs[1].units)},
                    'index': int(slider.val) - 1
                }
            elif slice_state == 'y':
                self.exp_data['slice'] = {
                    ys[0]: {'array': line.get_xdata(), 'unit': str(ys[1].units)},
                    zs[0]: {'array': line.get_ydata(), 'unit': str(zs[1].units)},
                    'index': int(slider.val) - 1
                }              
            return slider

    def replot(self):
        if self.current_data is not None:
            xs, ys, zs = self.current_data[:]
            name = ys[0] if zs is None else zs[0]
            self.fig_title = f"{self.dataset.metadata['location']} [{name}]"
            self.subtract_background()

    def set_slice(self, idx=None):
        if isinstance(idx, QtWidgets.QRadioButton):
            idx = self.slice_radio.id(idx)
        if idx is None:
            idx = self.slice_radio.checkedId()
        self.slice_state = idx
        self.update_slice()
        self.replot()

    def update_slice(self):
        if self.slice_state == 1:
            self.pyqt_imview.h_cross_section_widget.hide()
            self.pyqt_imview.v_cross_section_widget.hide()
            return
        if self.slice_state == 2:
            self.pyqt_imview.h_cross_section_widget.show()
            self.pyqt_imview.v_cross_section_widget.hide()

        elif self.slice_state == 3:
            self.pyqt_imview.h_cross_section_widget.hide()
            self.pyqt_imview.v_cross_section_widget.show()
        else:
            raise ValueError("Unknown Slice State: {}".format(self.slice_state))

    def update_line_by_line(self):
        if not self.line_backsub_btn.isChecked():
            self.x_line_backsub_btn.setDisabled(True)
            self.y_line_backsub_btn.setDisabled(True)
        else:
            self.x_line_backsub_btn.setDisabled(False)
            self.y_line_backsub_btn.setDisabled(False)
        self.replot()

    def subtract_background(self, idx=None):
        if self.current_data is None:
            return
        if isinstance(idx, QtWidgets.QRadioButton):
            idx = self.backsub_radio.id(idx)
        idx = idx or self.backsub_radio.checkedId()
        xs, ys, zs = self.current_data[:]
        line_by_line = self.line_backsub_btn.isChecked()
        if line_by_line:
            funcs = (lambda x: 0, np.min, np.max, np.mean, np.median,
                        lambda y, x=xs[1].magnitude: self._subtract_line(x, y))
            axis = self.line_backsub_radio.checkedId()
            z = self._subtract_line_by_line(np.copy(zs[1].magnitude), axis, funcs[idx])
            zs = [zs[0], z * zs[1].units]
        if idx == 1:
            if zs is None:
                ys = [ys[0], ys[1] - np.min(ys[1])]
            elif not line_by_line:
                zs = [zs[0], zs[1] - np.min(zs[1])]
        elif idx == 2:
            if zs is None:
                ys = [ys[0], ys[1] - np.max(ys[1])]
            elif not line_by_line:
                zs = [zs[0], zs[1] - np.max(zs[1])] 
        elif idx == 3:
            if zs is None:
                ys = [ys[0], ys[1] - np.mean(ys[1])]
            elif not line_by_line:
                zs = [zs[0], zs[1] - np.mean(zs[1])]
        elif idx == 4:
            if zs is None:
                ys = [ys[0], ys[1] - ys[1].units * np.median(ys[1])]
            elif not line_by_line:
                zs = [zs[0], zs[1] - zs[1].units * np.median(np.reshape(zs[1], (-1,1)))]
        elif idx == 5:
            if zs is None:
                slope, offset = np.polyfit(xs[1].magnitude, ys[1].magnitude, 1)
                ys = [ys[0], ys[1] -  ys[1].units * (slope * xs[1].magnitude + offset)]
            elif not line_by_line:
                X, Y = np.meshgrid(xs[1], ys[1], indexing='ij')
                x = np.reshape(X, (-1, 1))
                y = np.reshape(Y, (-1, 1))
                data = np.reshape(zs[1].magnitude, (-1, 1))
                z = np.column_stack((x, y, np.ones_like(x)))
                plane, _, _, _ = lstsq(z, data)
                zs = [zs[0], zs[1] - zs[1].units * (plane[0] * X + plane[1] * Y + plane[2])]
        #name = ys[0] if zs is None else zs[0]
        #self.fig_title = f"{self.dataset.metadata['location']} [{name}]"
        self.plot_arrays(xs, ys, zs=zs, title=self.fig_title)
        #self.fig.tight_layout()
        #self.fig.subplots_adjust(top=0.9, bottom=0.11)

    def export_mpl(self, dpi=300):
        name = self.fig_title.split('/')[-1].replace(' ', '')
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export matplotlib', name,
                                                    'PNG Image (*.png);;JPEG Image (*.jpg)')
        self.fig.savefig(path, dpi=dpi)

    def export_pg(self, width=1200):
        name = self.fig_title.split('/')[-1].replace(' ', '')
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export pyqtgraph', name,
                                                    'PNG Image (*.png);;JPEG Image (*.jpg)')
        plot = self.pyqt_plot.plotItem if self.pyqt_plot.isVisible() else self.pyqt_imview.scene
        exporter = pg.exporters.ImageExporter(plot)
        exporter.parameters()['width'] = width
        exporter.export(path)

    def export_data(self):
        if not self.exp_data:
            return
        name = self.fig_title.split('/')[-1].replace(' ', '')
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export current data', name,
                                                    'MAT File (*.mat);;HDF5 (*.h5)')
        if path.endswith('mat'):
            io.savemat(path, self.exp_data)
        elif path.endswith('h5'):
            with h5py.File(path) as df:
                set_h5_attrs(df, self.exp_data)


    def _subtract_line_by_line(self, zdata, axis, func):
        if axis: # y
            for i in range(zdata.shape[axis]):
                zdata[:,i] -= func(zdata[:,i])
        else: #x
            for i in range(zdata.shape[axis]):
                zdata[i,:] -= func(zdata[i,:])
        return zdata

    def _subtract_line(self, x, y):
        slope, offset = np.polyfit(x, y, 1)
        return y - (slope * x + offset)

class DataSetPlotter(PlotWidget):
    def __init__(self, dataset=None, parent=None, init_plot=False):
        super(DataSetPlotter, self).__init__(parent=parent)
        self.dataset = dataset
        self.default_dataset = ''
        self.dataset_state = {}
        self.arrays = None
        self.indep_vars = None
        arrays_widget = QtWidgets.QGroupBox('Arrays')
        arrays_layout = QtWidgets.QHBoxLayout(arrays_widget)
        arrays_widget.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.selector = QtWidgets.QComboBox()
        arrays_layout.addWidget(self.selector)
        self.option_layout.insertWidget(0, arrays_widget)
        self.selector.currentIndexChanged.connect(self.set_plot)

        self.xy_units_box = QtWidgets.QCheckBox('real x-y units')
        self.xy_units_box.setChecked(False)
        self.xy_units = QtWidgets.QLineEdit()
        self.xy_units.setText('Enter length unit')
        self.xy_units.setDisabled(True)
        xy_units_widget = QtWidgets.QWidget()
        xy_units_layout = QtWidgets.QVBoxLayout()
        xy_units_widget.setLayout(xy_units_layout)
        xy_units_layout.addWidget(self.xy_units_box)
        xy_units_layout.addWidget(self.xy_units)
        #arrays_layout.addWidget(self.xy_units_box)
        # arrays_layout.addWidget(self.xy_units)
        arrays_layout.addWidget(xy_units_widget)
        self.xy_units_box.stateChanged.connect(self.update_xy_units)
        self.xy_units.returnPressed.connect(self.update)

        for toggle in self.opt_checks.values():
            toggle.toggled.connect(self.update_dataset_state)

    def get_arrays(self):
        if self.dataset is None:
            return
        if '_scan_' in self.dataset.location:
            unit = self.xy_units.text() if self.xy_units_box.isChecked() else None
            self.arrays = scan_to_arrays(self.dataset, xy_unit=unit)
            self.indep_vars = ('x', 'y')
        elif '_td_cap_' in self.dataset.location:
            self.arrays = td_to_arrays(self.dataset)
            self.indep_vars = ('height',)
        last_text = self.selector.currentText()
        if not last_text:
            last_text = self.default_dataset
        self._disable_update = True
        self.selector.clear()
        items = []
        for name in self.arrays.keys():
            if name.lower() not in self.indep_vars:
                items.append(name)
                self.selector.addItem(name)
        self._disable_update = False
        self.selector.setCurrentIndex(0)
        self.set_plot(0)
        # if items:
        #     last_text = items[0]
        # try:
        #     self.selector.go_to_item(last_text)
        #     self.set_plot_from_name(last_text)
        # except ValueError:
        #     pass

    def set_plot(self, idx):
        if self._disable_update:
            return
        name = str(self.selector.itemText(idx))
        if not name:
            return
        self.set_plot_from_name(name)

    def set_plot_from_name(self, name):
        if len(self.indep_vars) == 1:
            xs = [self.indep_vars[0], self.arrays[self.indep_vars[0]]]
            ys = [name, self.arrays[name]]
            zs = None
        elif len(self.indep_vars) == 2:
            xs, ys = ([var, self.arrays[var]] for var in self.indep_vars)
            z = self.arrays[name]
            z[np.isnan(z)] = np.nanmin(z) * z.units
            zs = [name, z]
        title = ''
        if self.dataset is not None:
            title = f"{self.dataset.metadata['location']} [{name}]"
        self.current_data = [xs, ys, zs]
        self.plot_arrays(xs, ys, zs, title)
        self.subtract_background()
        if name in self.dataset_state:
            self.set_dataset_state(self.dataset_state[name])
        else:
            self.update_dataset_state()

    def update_xy_units(self):
        if not self.xy_units_box.isChecked():
            self.xy_units.setText('Enter length unit')
            self.xy_units.setDisabled(True)
        else:
            self.xy_units.setText('um')
            self.xy_units.setDisabled(False)
        idx = self.selector.currentIndex()
        self.update(idx=idx)
        self.selector.setCurrentIndex(idx)

    def update_dataset_state(self):
        self.dataset_state[str(self.selector.currentText())] = self.get_all_opts()

    def set_dataset_state(self, state):
        self.set_opts(state)

    def get_state(self):
        state = self.dataset_state.copy()
        state['default_dataset'] = self.selector.currentText()
        return state

    def set_state(self, state):
        self.default_dataset = state.pop('default_dataset', '')
        self.dataset_state = state

    def update(self, dataset=None, idx=0):
        self.dataset = dataset or self.dataset
        self.get_arrays()
        self.backsub_radio.button(0).setChecked(True)
        self.set_plot(idx)


class ImageView(pg.ImageView):
    def __init__(self, **kwargs):
        kwargs['view'] = pg.PlotItem(labels=kwargs.pop('labels', None))
        super().__init__(**kwargs)
        self.view.setAspectLocked(lock=True)
        self.view.invertY(False)
        self.set_histogram(True)
        histogram_action = QtWidgets.QAction('Histogram', self)
        histogram_action.setCheckable(True)
        histogram_action.triggered.connect(self.set_histogram)
        self.scene.contextMenu.append(histogram_action)
        self.ui.histogram.gradient.loadPreset('grey')

    def setLabels(self, xlabel='X', ylabel='Y', zlabel='Z'):
        self.view.setLabels(bottom=(xlabel,), left=(ylabel,))
        self.h_cross_section_widget.plotItem.setLabels(bottom=xlabel, left=zlabel)
        self.v_cross_section_widget.plotItem.setLabels(bottom=ylabel, left=zlabel)
        self.ui.histogram.item.axis.setLabel(text=zlabel)

    def set_histogram(self, visible):
        self.ui.histogram.setVisible(visible)
        self.ui.roiBtn.setVisible(False)
        self.ui.normGroup.setVisible(False)
        self.ui.menuBtn.setVisible(False)
        # self.getImageItem().setLookupTable(self.lut)

    def set_data(self, data):
        self.setImage(data)
        # self.getImageItem().setLookupTable(self.lut)

    def set_cmap(self, name):
        self.ui.histogram.gradient.loadPreset(name)


class CrosshairPlotWidget(pg.PlotWidget):
    crosshair_moved = QtCore.Signal(float, float)
    #hist_plot = None
    def __init__(self, parametric=False, *args, **kwargs):
        super(CrosshairPlotWidget, self).__init__(*args, **kwargs)
        self.scene().sigMouseClicked.connect(self.toggle_search)
        self.scene().sigMouseMoved.connect(self.handle_mouse_move)
        self.cross_section_enabled = False
        self.parametric = parametric
        self.search_mode = True
        self.label = None
        self.selected_point = None
        self.plotItem.showGrid(x=True, y=True, alpha=0.5)

    def set_data(self, data):
        if data is not None and len(data) > 0 and np.isfinite(data).all():
            self.clear()
            self.plot(data)

    def toggle_search(self, mouse_event):
        if mouse_event.double():
            if self.cross_section_enabled:
                self.hide_cross_hair()
            else:
                self.add_cross_hair()
        elif self.cross_section_enabled:
            self.search_mode = not self.search_mode
            if self.search_mode:
                self.handle_mouse_move(mouse_event.scenePos())

    def handle_mouse_move(self, mouse_event):
        if self.cross_section_enabled and self.search_mode:
            item = self.getPlotItem()
            vb = item.getViewBox()
            view_coords = vb.mapSceneToView(mouse_event)
            view_x, view_y = view_coords.x(), view_coords.y()

            best_guesses = []
            for data_item in item.items:
                if isinstance(data_item, pg.PlotDataItem):
                    xdata, ydata = data_item.xData, data_item.yData
                    index_distance = lambda i: (xdata[i]-view_x)**2 + (ydata[i] - view_y)**2
                    if self.parametric:
                        index = min(range(len(xdata)), key=index_distance)
                    else:
                        index = min(np.searchsorted(xdata, view_x), len(xdata)-1)
                        if index and xdata[index] - view_x > view_x - xdata[index - 1]:
                            index -= 1
                    pt_x, pt_y = xdata[index], ydata[index]
                    best_guesses.append(((pt_x, pt_y), index_distance(index), index))

            if not best_guesses:
                return

            (pt_x, pt_y), _, index = min(best_guesses, key=lambda x: x[1])
            self.selected_point = (pt_x, pt_y)
            self.v_line.setPos(pt_x)
            self.h_line.setPos(pt_y)
            self.label.setText("x={:.5f}, y={:.5f}".format(pt_x, pt_y))
            self.crosshair_moved.emit(pt_x, pt_y)
    #         self.update_hist(index)

    # def update_hist(self, index):
    #     if self.hist_plot is not None and self.hist_plot.isVisible():
    #         wvals = self.raw_iq_data[:, index]
    #         H, xs, ys = np.histogram2d(wvals.real, wvals.imag, bins=25, normed=True)
    #         self.hist_plot.setImage(H, pos=(xs[0], ys[0]), scale=(xs[1]-xs[0], ys[1]-ys[0]))

    def add_cross_hair(self):
        self.h_line = pg.InfiniteLine(angle=0, movable=False)
        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.addItem(self.h_line, ignoreBounds=False)
        self.addItem(self.v_line, ignoreBounds=False)
        if self.label is None:
            self.label = pg.LabelItem(justify="right")
            self.getPlotItem().layout.addItem(self.label, 4, 1)
        self.x_cross_index = 0
        self.y_cross_index = 0
        self.cross_section_enabled = True

    def hide_cross_hair(self):
        self.removeItem(self.h_line)
        self.removeItem(self.v_line)
        self.cross_section_enabled = False

class CrossSectionImageView(ImageView):
    #hist_plot = None
    def __init__(self, **kwargs):
        super(CrossSectionImageView, self).__init__(**kwargs)
        self.search_mode = False
        self.signals_connected = False
        try:
            self.connect_signal()
        except RuntimeError:
            logger.warn('Scene not set up, cross section signals not connected')
        self.angle = 0
        self.y_cross_index = 0
        self.x_cross_index = 0
        self.h_cross_section_widget = CrosshairPlotWidget()
        self.h_cross_section_widget.add_cross_hair()
        self.h_cross_section_widget.search_mode = False
        self.h_cross_section_widget_data = self.h_cross_section_widget.plot([0,0])
        self.h_line = pg.InfiniteLine(pos=0, angle=0, movable=False)
        self.view.addItem(self.h_line, ignoreBounds=False)

        self.v_cross_section_widget = CrosshairPlotWidget()
        self.v_cross_section_widget.add_cross_hair()
        self.v_cross_section_widget.search_mode = False
        self.v_cross_section_widget_data = self.v_cross_section_widget.plot([0,0])
        self.v_line = pg.InfiniteLine(pos=0, angle=90, movable=False)
        self.view.addItem(self.v_line, ignoreBounds=False)

        self.h_cross_section_widget.crosshair_moved.connect(lambda x, _: self.set_position(x=x))
        self.v_cross_section_widget.crosshair_moved.connect(lambda y, _: self.set_position(y=y))

        self.text_item = pg.LabelItem(justify="right")
        self.view.layout.addItem(self.text_item, 4, 1)

    def setImage(self, *args, **kwargs):
        if 'pos' in kwargs:
            self._x0, self._y0 = kwargs['pos']
        else:
            self._x0, self._y0 = 0, 0
        if 'scale' in kwargs:
            self._xscale, self._yscale = kwargs['scale']
        else:
            self._xscale, self._yscale = 1, 1

        # Adjust to make pixel centers align on ticks
        self._x0 -= self._xscale/2.0
        self._y0 -= self._yscale/2.0
        if 'pos' in kwargs:
            kwargs['pos'] = self._x0, self._y0

        if self.imageItem.image is not None:
            (min_x, max_x), (min_y, max_y) = self.imageItem.getViewBox().viewRange()
            mid_x, mid_y = (max_x + min_x)/2., (max_y + min_y)/2.
        else:
            mid_x, mid_y = 0, 0

        self.h_line.setPos(mid_y)
        self.v_line.setPos(mid_x)

        super().setImage(*args, **kwargs)
        self.set_position()

    def connect_signal(self):
        """This can only be run after the item has been embedded in a scene"""
        if self.imageItem.scene() is None:
            raise RuntimeError('Signal can only be connected after it has been embedded in a scene.')
        self.imageItem.scene().sigMouseClicked.connect(self.toggle_search)
        self.imageItem.scene().sigMouseMoved.connect(self.handle_mouse_move)
        self.timeLine.sigPositionChanged.connect(self.update_cross_section)
        self.signals_connected = True

    def toggle_search(self, mouse_event):
        if mouse_event.double():
            return
        self.search_mode = not self.search_mode
        if self.search_mode:
            self.handle_mouse_move(mouse_event.scenePos())

    def handle_mouse_move(self, mouse_event):
        if self.search_mode:
            view_coords = self.imageItem.getViewBox().mapSceneToView(mouse_event)
            view_x, view_y = view_coords.x(), view_coords.y()
            self.set_position(view_x, view_y)

    def set_position(self, x=None, y=None):
        if x is None:
            x = self.v_line.getXPos()
        if y is None:
            y = self.h_line.getYPos()

        item_coords = self.imageItem.getViewBox().mapFromViewToItem(self.imageItem, QtCore.QPointF(x, y))
        item_x, item_y = item_coords.x(), item_coords.y()
        max_x, max_y = self.imageItem.image.shape

        item_x = self.x_cross_index = max(min(int(item_x), max_x-1), 0)
        item_y = self.y_cross_index = max(min(int(item_y), max_y-1), 0)

        view_coords = self.imageItem.getViewBox().mapFromItemToView(self.imageItem, QtCore.QPointF(item_x+.5, item_y+.5))
        x, y = view_coords.x(), view_coords.y()

        self.v_line.setPos(x)
        self.h_line.setPos(y)
        z_val = self.imageItem.image[self.x_cross_index, self.y_cross_index]
        # self.update_hist()
        self.update_cross_section()
        self.text_item.setText('x={:.5f}, y={:.5f}, z={:.5f}'.format(x, y, z_val))

    # def update_hist(self):
    #     if self.hist_plot is not None and self.hist_plot.isVisible():
    #         wvals = self.raw_iq_data[:, self.x_cross_index, self.y_cross_index]
    #         H, xs, ys = np.histogram2d(wvals.real, wvals.imag, bins=25, normed=True)
    #         self.hist_plot.setImage(H, pos=(xs[0], ys[0]), scale=(xs[1]-xs[0], ys[1]-ys[0]))

    def update_cross_section(self):
        zdata = self.imageItem.image
        nx, ny = zdata.shape
        x0, y0, xscale, yscale = self._x0, self._y0, self._xscale, self._yscale
        xdata = np.linspace(x0, x0+(xscale*(nx-1)), nx)
        ydata = np.linspace(y0, y0+(yscale*(ny-1)), ny)
        zval = zdata[self.x_cross_index, self.y_cross_index]
        self.h_cross_section_widget_data.setData(xdata, zdata[:, self.y_cross_index])
        self.h_cross_section_widget.v_line.setPos(xdata[self.x_cross_index])
        self.h_cross_section_widget.h_line.setPos(zval)
        self.v_cross_section_widget_data.setData(ydata, zdata[self.x_cross_index, :])
        self.v_cross_section_widget.v_line.setPos(ydata[self.y_cross_index])
        self.v_cross_section_widget.h_line.setPos(zval)