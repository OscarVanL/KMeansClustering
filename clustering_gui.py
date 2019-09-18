import random

from k_means import KMeans
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog, QErrorMessage, QVBoxLayout, QLabel
from PyQt5.QtCore import pyqtSlot
from PyQt5 import uic
import os
import sys

main_interface_file = os.path.join('layout', 'main_interface.ui')  # OS-safe path slashes

class ClusteringGui(QMainWindow):
    layout: QVBoxLayout  # Main Layout in which everything else is contained
    browse_btn: QPushButton  # Load Dataset button
    step_btn: QPushButton  # Next button
    dimensions_label: QLabel  # Filled with vector dimensions found from dataset
    recommended_k_label: QLabel  # Filled with recommended K value as calculated using SSE / Elbow method
    plot_canvas: FigureCanvas  # Canvas containing matplotlib plot

    kmeans: KMeans
    figure: plt.Figure

    def __init__(self):
        self.kmeans = KMeans(dimensions=2)
        super(ClusteringGui, self).__init__()
        uic.loadUi(main_interface_file, self)

        self.browse_btn = self.findChild(QPushButton, 'browse_button')
        self.browse_btn.clicked.connect(self.on_browse_click)
        self.step_btn = self.findChild(QPushButton, 'step_button')
        self.step_btn.clicked.connect(self.on_step_click)
        self.layout = self.findChild(QVBoxLayout, 'layout')
        self.dimensions_label = self.findChild(QLabel, 'dimensions_label')

        self.show()

    @pyqtSlot()
    def on_browse_click(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "CSV Files (*.csv)")
        try:
            self.kmeans.open_dataset(filepath=path)
            self.dimensions_label.setText("Dimensions: {}".format(self.kmeans.dimensions))
        except FileNotFoundError:
            print("Exception")
            error_dialog = QErrorMessage()
            error_dialog.showMessage("File Not Found, try again")
            error_dialog.exec()

        self.add_matplotlib_canvas()

    @pyqtSlot()
    def on_step_click(self):
        self.kmeans.next_step()
        self.update_matplotlib()


    def add_matplotlib_canvas(self):
        self.figure = plt.figure()
        self.plot_canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.plot_canvas)

        self.figure.clear()

        ax = self.figure.add_subplot(111)

        x = [point[0] for point in self.kmeans.vectors]
        y = [point[1] for point in self.kmeans.vectors]
        ax.scatter(x, y, alpha=0.8)

        self.plot_canvas.draw()

    def update_matplotlib(self):
        self.figure.clear()

        ax = self.figure.add_subplot(111)

        for cluster in self.kmeans.clusters:
            x = [point[0] for point in cluster]
            y = [point[1] for point in cluster]
            ax.scatter(x, y, alpha=0.8)

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
