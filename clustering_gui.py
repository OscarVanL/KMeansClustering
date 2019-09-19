from k_means import KMeans

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog, QErrorMessage, QVBoxLayout, QLabel, QSpinBox, QWidget, QGridLayout
from PyQt5.QtCore import pyqtSlot
from PyQt5 import uic
import os
import sys

main_interface_file = os.path.join('layout', 'main_interface.ui')  # OS-safe path slashes


class ClusteringGui(QMainWindow):
    # ui
    layout: QVBoxLayout  # Main Layout in which everything else is contained
    browse_btn: QPushButton  # Load Dataset button
    k_selector: QSpinBox  # Selector for k value
    repetitions_selector: QSpinBox  # Select how many times K-Means is ran per dataset
    run_btn: QPushButton  # Run button
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
        self.repetitions_selector = self.findChild(QSpinBox, 'k_repetitions_selector')
        self.repetitions_selector.valueChanged.connect(self.on_set_repetitions)
        self.run_btn = self.findChild(QPushButton, 'run_button')
        self.run_btn.clicked.connect(self.on_run_click)
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
            self.setWindowTitle('K-Means Clustering: {}'.format(path))
        except FileNotFoundError:
            print("Exception")
            error_dialog = QErrorMessage()
            error_dialog.showMessage("File Not Found, try again")
            error_dialog.exec()

    @pyqtSlot()
    def on_show_elbow(self):
        sse = []
        # Calculate SSE for values of K ranging from 1 to 10.
        for i in range(1,10):
            print("Calculating for K =", i)
            self.kmeans.update_k(i)
            # Cluster and update centroids 3 times for each K value
            self.kmeans.cluster_points()
            self.kmeans.cluster_points()
            self.kmeans.cluster_points()
            sse.append(self.kmeans.calculate_sse())

        self.elbow_chart = ElbowChartGui()
        self.elbow_chart.plot(sse)

    # Handles K value being changed
    @pyqtSlot()
    def on_update_k(self):
        self.kmeans.update_k(self.k_selector.value())
        # If there's data being displayed, update it now
        if self.kmeans.data_displayed:
            self.update_matplotlib()

    @pyqtSlot()
    def on_run_click(self):
        self.kmeans.run()
        self.update_matplotlib()

    @pyqtSlot()
    def on_set_repetitions(self):
        self.kmeans.repetitions = self.repetitions_selector.value()

    # Handles 'Next' button being pressed
    @pyqtSlot()
    def on_step_click(self):
        self.kmeans.step()
        self.update_matplotlib()

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

# A Gui for displaying the Elbow Chart generated for a K-Means dataset
# This hopes to inform the user as to what K value to use
class ElbowChartGui(QWidget):

    def __init__(self):
        super(ElbowChartGui, self).__init__()
        self.layout = QGridLayout(self)
        self.figure = plt.figure()
        self.plot_canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.plot_canvas)
        self.setLayout(self.layout)

    def plot(self, sse: [int]):
        ax = self.figure.add_subplot(1, 1, 1)
        ax.plot([1,2,3,4,5,6,7,8,9], sse)
        self.plot_canvas.draw()
        self.show()
        print(sse)



def main():
    app = QApplication(sys.argv)
    window = ClusteringGui()
    window.show()
    app.exec_()


# Ensures exceptions from PyQt Slots are passed to stdout/stderr
def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


sys.excepthook = except_hook
main()
