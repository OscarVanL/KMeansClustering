from k_means import KMeans
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
from PyQt5.QtCore import pyqtSlot
from PyQt5 import uic
import os
import sys

main_interface_file = os.path.join('layout', 'main_interface.ui')  # OS-safe path slashes

class ClusteringGui(QMainWindow):
    kmeans = None

    def __init__(self):
        self.kmeans = KMeans(dimensions=2)
        super(ClusteringGui, self).__init__()
        uic.loadUi(main_interface_file, self)
        self.browse = self.findChild(QPushButton, "browse_button")
        self.browse.clicked.connect(self.on_browse_click)

        self.show()

    @pyqtSlot()
    def on_browse_click(self):
        print("Button clicked")


def main():
    app = QApplication(sys.argv)
    window = ClusteringGui()
    window.show()
    sys.exit(app.exec_())
    pass


main()
