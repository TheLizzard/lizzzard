# This code is a terrible way of checking if we are using python 2 or 3
#   but it works (on my computer :D)
try:
    import Tkinter
    PYTHON = 2
except ImportError as error:
    PYTHON = 2
    if "ModuleNotFoundError" in str(type(error)):
        PYTHON = 3

if PYTHON == 2:
    bytes, str, bytes2 = str, unicode, bytes
else:
    bytes2 = str