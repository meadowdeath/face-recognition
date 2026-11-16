"""Shared flat widget palettes. No screen reconstruction on appearance changes."""

THEMES = {
    "dark": dict(background="#100D18", sidebar="#191325", card="#241B35", accent="#A78BFA",
        button="#7C3AED", pressed="#6D28D9", border="#3D3054", text="#F5F3FF", secondary="#B6ABC9",
        success="#4ADE80", error="#F87171", warning="#FBBF24", information="#60A5FA"),
    "light": dict(background="#F8F7FC", sidebar="#FFFFFF", card="#FFFFFF", accent="#6D28D9",
        button="#7C3AED", pressed="#5B21B6", border="#E4DDF0", text="#211831", secondary="#625770",
        success="#15803D", error="#B91C1C", warning="#A16207", information="#1D4ED8"),
}


def stylesheet(name):
    c = THEMES[name]
    return f"""
    QWidget {{ background: {c['background']}; color: {c['text']}; font-family: 'Segoe UI', 'DejaVu Sans'; font-size: 12px; }}
    QFrame#sidebar, QFrame#status {{ background: {c['sidebar']}; }}
    QFrame#card, QFrame#preview {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 10px; }}
    QLabel {{ background: transparent; border: none; }}
    QLabel#title {{ font-size: 23px; font-weight: 600; }}
    QLabel#brand {{ font-size: 27px; font-weight: 700; }}
    QLabel#muted {{ color: {c['secondary']}; }}
    QLabel#scan {{ color: {c['accent']}; font-size: 45px; }}
    QLabel#success {{ color: {c['success']}; }}
    QLabel#error {{ color: {c['error']}; }}
    QLabel#warning {{ color: {c['warning']}; }}
    QLabel#information {{ color: {c['information']}; }}
    QPushButton {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 5px 8px; min-height: 32px; }}
    QPushButton:hover {{ border-color: {c['accent']}; }}
    QPushButton:pressed {{ background: {c['pressed']}; color: white; }}
    QPushButton:checked {{ border: 2px solid {c['accent']}; color: {c['accent']}; }}
    QPushButton#primary {{ background: {c['button']}; color: white; border: none; }}
    QPushButton#primary:pressed {{ background: {c['pressed']}; }}
    QPushButton#destructive {{ color: {c['error']}; border-color: {c['error']}; }}
    QPushButton#nav {{ text-align: left; padding-left: 16px; }}
    QPushButton#nav:checked {{ background: {c['accent']}; color: {c['sidebar']}; }}
    QLineEdit, QComboBox, QDateEdit {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 6px; padding: 5px; min-height: 30px; }}
    QLineEdit[pinActive="true"] {{ border: 2px solid {c['accent']}; }}
    QTableWidget {{ background: {c['card']}; gridline-color: {c['border']}; border: 1px solid {c['border']}; selection-background-color: {c['button']}; selection-color: white; }}
    QTableWidget::item {{ padding: 5px; }}
    QHeaderView::section {{ background: {c['sidebar']}; color: {c['secondary']}; border: none; padding: 8px; }}
    QScrollArea {{ border: none; }}
    """
