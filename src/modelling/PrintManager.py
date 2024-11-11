# src/modelling/PrintManager.py

# PrintManager class made for easily managing print statements, and whether/where
# to print or not to print. This proves especially helpful when running Habrok

import sys

class PrintManager:
    """
    Helps with managing print statements, and also where to print them.
    Especially when running Habrok, which contains no ordinary out-stream,
    it is nice to keep track of model training by passing the logging to
    a .txt file. This class makes that possible. In addition, it can also
    be easily turned on/off, so printing can be managed through one boolean.
    """
    def __init__(self, filename: str, mode: str, bool: bool):
        """
        Initialize object

        :param filename: name of the file to print to
        :param mode: mode to open the file in (usually just 'a' for append)
        :param bool: whether to print to file, at all, or not
        """
        self.filename = filename
        self.mode = mode
        self.bool = bool
        self.file = None
        self.stdout_original = sys.stdout

    def __enter__(self):
        """
        Enter method for context manager, which opens the file. In practice,
        this method is called when the 'with' statement is used. When bool is
        set to False, the original stdout is used/restored and no file is opened
        """
        if self.bool:
            self.file = open(self.filename, self.mode)
            sys.stdout = self.file
        else:
            sys.stdout = self.stdout_original
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        When the context manager is exited, the file is closed and the original
        stdout is restored. If an exception occurred, this method will print it
        """
        if self.file is not None:
            self.file.close()
            sys.stdout = sys.__stdout__
        if exc_type is not None:
            print(f"An exception of type {exc_type} occurred with value {exc_val}")
            print("Traceback:", exc_tb)
            return False