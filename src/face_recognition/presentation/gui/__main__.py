"""Run on the native Qt backend; EGLFS is selected by the operator environment."""

import argparse
import sys

from face_recognition.presentation.cli.live_options import add_live_arguments, validate_live_arguments


def main(argv=None):
    parser = argparse.ArgumentParser(description="Quiosco Face ID; reconocimiento en vivo opcional con --model-dir")
    parser.add_argument("--windowed", action="store_true", help="Usar ventana de desarrollo de 800 × 480")
    add_live_arguments(parser)
    args = parser.parse_args(argv)
    validate_live_arguments(parser, args)
    try:
        from PyQt5.QtWidgets import QApplication
        from .window import KioskWindow
    except ImportError:
        parser.exit(1, "PyQt5 is required for the GUI. Install the optional [gui] dependencies.\n")
    app = QApplication(sys.argv[:1])
    window = KioskWindow(live_mode=args.model_dir is not None, debug=args.debug)
    integration = None
    if args.model_dir is not None:
        from .recognition import RecognitionIntegration
        integration = RecognitionIntegration(window, args)
        app.aboutToQuit.connect(integration.finalize)
    if not args.windowed and app.platformName() == "eglfs":
        window.showFullScreen()
    else:
        window.show()
    try:
        return app.exec_()
    finally:
        if integration is not None:
            integration.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
