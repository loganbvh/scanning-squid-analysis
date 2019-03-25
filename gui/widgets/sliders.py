from ..qt import Qt, QtCore, QtWidgets, QtGui

class LabeledSlider(QtWidgets.QWidget):
    def __init__(self, minimum, maximum, init, interval=1, orientation=Qt.Horizontal,
            labels=None, parent=None):
        super().__init__(parent=parent)

        levels = range(minimum, maximum+interval, interval)
        if labels is not None:
            if not isinstance(labels, (tuple, list)):
                raise Exception("<labels> is a list or tuple.")
            if len(labels) != len(levels):
                raise Exception("Size of <labels> doesn't match levels.")
            self.levels=list(zip(levels,labels))
        else:
            self.levels=list(zip(levels,map(str,levels)))

        if orientation == Qt.Horizontal:
            self.layout=QtWidgets.QVBoxLayout(self)
        elif orientation == Qt.Vertical:
            self.layout=QtWidgets.QHBoxLayout(self)
        else:
            raise Exception("<orientation> wrong.")

        # gives some space to print labels
        self.left_margin = 10
        self.top_margin = 0
        self.right_margin = 10
        self.bottom_margin = 10

        self.layout.setContentsMargins(self.left_margin,self.top_margin,
                self.right_margin,self.bottom_margin)

        self.slider = QtWidgets.QSlider(orientation, self)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setValue(init)
        if orientation == Qt.Horizontal:
            self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        else:
            self.slider.setTickPosition(QtWidgets.QSlider.TicksLeft)
        self.slider.setTickInterval(interval)
        self.slider.setSingleStep(1)
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)

        self.layout.addWidget(self.slider)

    def paintEvent(self, e):
        super().paintEvent(e)
        style = self.slider.style()
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(Qt.black,1))
        st_slider = QtWidgets.QStyleOptionSlider()
        st_slider.initFrom(self.slider)
        st_slider.orientation = self.slider.orientation()

        length = style.pixelMetric(QtWidgets.QStyle.PM_SliderLength, st_slider, self.slider)
        available = style.pixelMetric(QtWidgets.QStyle.PM_SliderSpaceAvailable, st_slider, self.slider)

        for v, v_str in self.levels:
            # get the size of the label
            rect = painter.drawText(QtCore.QRect(), Qt.TextDontPrint, v_str)

            if self.slider.orientation() == Qt.Horizontal:
                # I assume the offset is half the length of slider, therefore
                # + length//2
                x_loc = QtWidgets.QStyle.sliderPositionFromValue(self.slider.minimum(),
                        self.slider.maximum(), v, available)+length//2

                # left bound of the text = center - half of text width + L_margin
                left = x_loc-rect.width()//2+self.left_margin
                bottom = self.rect().bottom()

                # enlarge margins if clipping
                if v == self.slider.minimum():
                    if left <= 0:
                        self.left_margin=rect.width()//2-x_loc
                    if self.bottom_margin <= rect.height():
                        self.bottom_margin=rect.height()

                    self.layout.setContentsMargins(self.left_margin,
                            self.top_margin, self.right_margin,
                            self.bottom_margin)

                if v == self.slider.maximum() and rect.width()//2 >= self.right_margin:
                    self.right_margin=rect.width()//2
                    self.layout.setContentsMargins(self.left_margin,
                            self.top_margin, self.right_margin,
                            self.bottom_margin)

            else:
                y_loc = QtWidgets.QStyle.sliderPositionFromValue(self.slider.minimum(),
                        self.slider.maximum(), v, available, upsideDown=True)

                bottom = y_loc+length//2+rect.height()//2+self.top_margin-3
                # there is a 3 px offset that I can't attribute to any metric

                left = self.left_margin-rect.width()
                if left <= 0:
                    self.left_margin=rect.width()+2
                    self.layout.setContentsMargins(self.left_margin,
                            self.top_margin, self.right_margin,
                            self.bottom_margin)

            pos = QtCore.QPoint(left, bottom)
            painter.drawText(pos, v_str)
        return

class SliderWidget(QtWidgets.QWidget):
    def __init__(self, min, max, init, interval, parent=None):
        super().__init__(parent=parent)
        self.angle_box = QtWidgets.QSpinBox(self)
        self.slider = LabeledSlider(min, max, init, interval=interval)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.angle_box)
        layout.addWidget(self.slider)
        self.slider.slider.valueChanged.connect(self.change_angle)
        self.angle_box.setMinimum(min)
        self.angle_box.setMaximum(max)
        self.angle_box.setValue(init)
        self.angle_box.setKeyboardTracking(False)
        self.angle_box.valueChanged.connect(self.change_angle)
        self.change_angle(init)
        sp = self.sizePolicy()
        sp.setVerticalPolicy(QtWidgets.QSizePolicy.Minimum)
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.MinimumExpanding)
        self.setSizePolicy(sp)

    def change_angle(self, val):
        self.angle_box.setValue(val)
        self.slider.slider.setValue(val)