import tkinter as tk

from interface.main_menu import SnowballGUI
from core.system.logger import SnowballLogger

def main():
    logger = SnowballLogger()
    root = tk.Tk()
    app = SnowballGUI(root, snowball_ai=None, logger=logger)
    root.mainloop()

if __name__ == "__main__":
    main()
