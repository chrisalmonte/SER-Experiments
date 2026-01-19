import os
import inspect
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
        self.properties = []
        self.epoch = []
    
    def log_message(self, message: str):
        self.messages.append(f"{datetime.now().strftime('%Y_%m_%d-%H:%M:%S')}: {message}")
        return message
    
    def log_property(self, key: str, value):
        property = f"{key}: {self.str_property(value)}"
        self.properties.append(property)
        return property

    def log_properties(self, set_name:str ,properties: dict):
        self.properties.append("\n" + set_name + ":\n")
        for key, value in properties.items():
            self.properties.append(f"{key}: {self.str_property(value)}")
        self.properties.append('\n')

    def log_epoch(self, epoch, properties: dict):
        self.epoch.append('\nEpoch num. ' + str(epoch))
        for key, value in properties.items():
            self.epoch.append(f"{key}: {value}")
        self.epoch.append('\n')

    def save(self):
        file_path = f"{self.path}/{self.prefix}_{datetime.now().strftime('%Y_%m_%d-%H%M%S')}.txt"       
        with open(file_path, 'w', encoding="utf-8") as file:
            file.write("Log saved on: ")
            file.write(datetime.now().strftime("%b %d %Y - %H:%M:%S") + "\n\n")
            if self.properties:
                file.write("-----Properties-----\n\n")
                file.write("\n".join(self.properties) + "\n\n")
            if self.messages:
                file.write("-----History-----\n\n")
                file.write("\n".join(self.messages) + "\n\n")
            if self.epoch:
                file.write("-----Training-----\n\n")
                file.write("\n".join(self.epoch) + "\n\n")
    
    @staticmethod
    def str_property(property):
        if inspect.isfunction(property) or inspect.ismethod(property):
            return property.__name__
        if callable(property):
            return property.__class__.__name__
        return str(property)

    