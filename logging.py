import os
from datetime import datetime

class Log:
    def __init__(self, path: str, prefix: str="log_"):
        #Vaidate path
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except:
                raise OSError("The directory could not be created.")
        self.prefix = prefix
        self.path = path
        self.messages = []
        self.properties = {}
    
    def log_message(self, message: str):
        self.messages.append(message)
    
    def log_property(self, key: str, value):
        self.properties[key] = value

    def log_properties(self, properties: dict):
        for key, value in properties.items():
            self.properties[key] = value

    def save(self):
        file_path = f"{self.path}/{self.prefix}_{datetime.now().strftime('%Y_%m_%d-%H%M%S')}.txt"       
        with open(file_path, 'w', encoding="utf-8") as file:
            file.write(datetime.now().strftime("%b %d %Y - %H:%M:%S") + "\n")
            file.write("History" + "\n")            
            file.write("\n".join(self.messages) + "\n\n")
            file.write("Properties" + "\n")
            for key, value in self.properties.items():
                file.write(f"{key}: {value}\n")
    