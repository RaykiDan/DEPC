import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms'
import sys
from PyQt5.QtWidgets import QApplication
from view.main_window import MainWindow
from model.stream_all import start_streaming
import threading

if __name__ == "__main__":
    # Jalankan streaming di thread terpisah
    streaming_thread = threading.Thread(target=start_streaming, daemon=True)
    streaming_thread.start()

    # Jalankan aplikasi utama
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())