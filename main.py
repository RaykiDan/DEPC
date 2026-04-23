import sys

from PyQt5.QtWidgets import QApplication
from qfluentwidgets import setTheme, Theme

from app.window import MainApp


def main():
    app = QApplication(sys.argv)
    setTheme(Theme.DARK)

    win = MainApp()
    win.setWindowTitle("Depth Estimation Performance on Mobile Robot")
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()