import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class window(QWidget):
    def __init__(self, parent=None):
        super(window, self).__init__(parent)
        self.resize(500, 500)
        self.setWindowTitle("Client")

        self.label = QLabel(self)
        self.label.setText("Federated Client")
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.move(0, 0)

        b2 = QPushButton(self)
        b2.setText("Button2")
        b2.move(50, 50)

        le = QLineEdit(self)
        le.setObjectName("host")
        le.setText("Host")
        le.resize(500, 60)
        le.move(10, 10)


def showdialog():
    dlg = QDialog()
    b1 = QPushButton("ok", dlg)
    b1.move(50, 50)
    b1.clicked.connect(showdialog)
    dlg.setWindowTitle("Dialog")
    dlg.setWindowModality(Qt.ApplicationModal)
    dlg.exec_()


def main():
    app = QApplication(sys.argv)

    showdialog()

    ex = window()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
