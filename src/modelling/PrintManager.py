import sys

class PrintManager:
    def __init__(self, filename, mode, bool):
        self.filename = filename
        self.mode = mode
        self.bool = bool
        self.file = None
        self.stdout_original = sys.stdout

    def __enter__(self):
        if self.bool:
            self.file = open(self.filename, self.mode)
            sys.stdout = self.file
        else:
            sys.stdout = self.stdout_original
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()
            sys.stdout = sys.__stdout__
        if exc_type is not None:
            print(f"An exception of type {exc_type} occurred with value {exc_val}")
            print("Traceback:", exc_tb)
            return False