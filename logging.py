import os
from datetime import datetime

class Log:
    def __init__(self, path: str):
        #Validar directorio de salida
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except:
                raise OSError("The directory could not be created.")
        self.path = path

    def save(self, filename: str):
        path = "%s%s_%s.txt" % (self.path, filename, datetime.now().strftime("%Y_%m_%d-%H%M%S"))        
        with open(path, 'w', encoding="utf-8") as file:
            file.write(datetime.now().strftime("%b %d %Y - %H:%M:%S") + "\n")
    