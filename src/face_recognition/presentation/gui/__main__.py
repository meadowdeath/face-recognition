"""Run on the native Qt backend; EGLFS is selected by the operator environment."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interfaz de demostración del quiosco Face ID")
    parser.add_argument("--windowed", action="store_true", help="Usar ventana de desarrollo de 800 × 480")
    args = parser.parse_args(argv)
    try:
        from PyQt5.QtWidgets import QApplication
        from .window import KioskWindow
    except ImportError:
        parser.exit(1, "PyQt5 is required for the GUI. Install the optional [gui] dependencies.\n")
    app = QApplication(sys.argv[:1])
    window = KioskWindow()
    if not args.windowed and app.platformName() == "eglfs":
        window.showFullScreen()
    else:
        window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
