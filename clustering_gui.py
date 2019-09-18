from k_means import KMeans

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog, QErrorMessage, QVBoxLayout, QLabel, QSpinBox
from PyQt5.QtCore import pyqtSlot
from PyQt5 import uic
import os
import sys

main_interface_file = os.path.join('layout', 'main_interface.ui')  # OS-safe path slashes


class ClusteringGui(QMainWindow):
    layout: QVBoxLayout  # Main Layout in which everything else is contained
    browse_btn: QPushButton  # Load Dataset button
    k_selector: QSpinBox  # Selector for k value
    step_btn: QPushButton  # Next button
    dimensions_label: QLabel  # Filled with vector dimensions found from dataset
    recommended_k_label: QLabel  # Filled with recommended K value as calculated using SSE / Elbow method
    plot_canvas: FigureCanvas  # Canvas containing matplotlib plot

    kmeans: KMeans
    figure: plt.Figure

    def __init__(self):
        self.kmeans = KMeans()
        super(ClusteringGui, self).__init__()
        uic.loadUi(main_interface_file, self)

        self.browse_btn = self.findChild(QPushButton, 'browse_button')
        self.browse_btn.clicked.connect(self.on_browse_click)
        self.k_selector = self.findChild(QSpinBox, 'k_val_selector')
        self.k_selector.valueChanged.connect(self.on_update_k)
        self.step_btn = self.findChild(QPushButton, 'step_button')
        self.step_btn.clicked.connect(self.on_step_click)
        self.elbow_btn = self.findChild(QPushButton, 'elbow_chart_button')
        self.elbow_btn.clicked.connect(self.on_show_elbow)
        self.layout = self.findChild(QVBoxLayout, 'layout')
        self.dimensions_label = self.findChild(QLabel, 'dimensions_label')

        self.show()

    @pyqtSlot()
    def on_browse_click(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "CSV Files (*.csv)")
        try:
            self.kmeans.open_dataset(filepath=path)
            self.dimensions_label.setText("Dimensions: {}".format(self.kmeans.dimensions))
            self.add_matplotlib_canvas()
        except FileNotFoundError:
            print("Exception")
            error_dialog = QErrorMessage()
            error_dialog.showMessage("File Not Found, try again")
            error_dialog.exec()

    # Handles K value being changed
    @pyqtSlot()
    def on_update_k(self):
        self.kmeans.update_k(self.k_selector.value())
        # If there's data being displayed, update it now
        if self.kmeans.data_displayed:
            self.update_matplotlib()

    # Handles 'Next' button being pressed
    @pyqtSlot()
    def on_step_click(self):
        self.kmeans.next_step()
        self.update_matplotlib()

    @pyqtSlot()
    def on_show_elbow(self):
        print(self.kmeans.calculate_sse())

    # Adds the matplotlib canvas to the UI
    def add_matplotlib_canvas(self):
        try:
            self.figure.clear()
        except AttributeError:
            self.figure = plt.figure()
            self.plot_canvas = FigureCanvas(self.figure)
            self.layout.addWidget(self.plot_canvas)

        ax = self.figure.add_subplot(1, 1, 1)

        x = [point[0] for point in self.kmeans.vectors]
        y = [point[1] for point in self.kmeans.vectors]
        ax.scatter(x, y, alpha=0.8)

        self.plot_canvas.draw()

    # Updates the matplotlib UI
    def update_matplotlib(self):
        self.figure.clear()

        ax = self.figure.add_subplot(1, 1, 1)

        for cluster in self.kmeans.clusters:
            x = [point[0] for point in cluster]
            y = [point[1] for point in cluster]
            ax.scatter(x, y, alpha=0.5)

        for centroid in self.kmeans.centroids:
            x, y = centroid[0], centroid[1]
            ax.scatter(x, y, c='k', alpha=0.8)

        self.plot_canvas.draw()


def main():
    app = QApplication(sys.argv)
    window = ClusteringGui()
    window.show()
    app.exec_()


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


sys.excepthook = except_hook
main()
