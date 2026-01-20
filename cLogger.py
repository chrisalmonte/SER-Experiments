import os
import pickle
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
        self.timings = []
        self.tracked_time_start = None
        self.last_save = None
    
    def log_message(self, message: str, show: bool=False):
        self.messages.append(f"{datetime.now().strftime('%Y_%m_%d-%H:%M:%S')}: {message}")
        if show:
            print(message)
    
    def log_property(self, key: str, value, show: bool=False):
        property = f"{key}: {self.str_property(value)}"
        self.properties.append(property)
        if show:
            print(property)

    def log_properties(self, set_name:str ,properties: dict):
        self.properties.append(' ')
        self.properties.append(set_name + ":\n")
        for key, value in properties.items():
            self.properties.append(f"{key}: {self.str_property(value)}")
        self.properties.append(' ')

    def log_epoch(self, epoch, properties: dict):
        self.properties.append(' ')
        self.epoch.append(f'Epoch num. {epoch}:' + '\n')
        for key, value in properties.items():
            self.epoch.append(f"{key}: {value}")
        self.properties.append(' ')

    def save(self, overwrite: bool=True):
        if overwrite and self.last_save:
            file_path = self.last_save
        else:
            file_path = f"{self.path}/{self.prefix}_{datetime.now().strftime('%Y_%m_%d-%H%M%S')}.pkl"       
            self.last_save = file_path
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    def save_txt(self):
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
            if self.timings:
                file.write("-----Timings-----\n\n")
                file.write("\n".join(self.timings) + "\n\n")
        
    def track_time(self, track: bool, message: str=""):
        if not track:
            self.tracked_time_start = None
        else:
            if self.tracked_time_start:
                self.log_message("Time tracking reset.", show=True)
            self.tracked_time_start = datetime.now()
            self.log_message(message if message else "Time tracking started.", show=True)            
    
    def log_elapsed_time(self, message: str="", reset_timer: bool=False):        
        if not self.tracked_time_start:
            raise ValueError("Time tracking has not been started. Call track_time() first.")
        else:
            elapsed_time = datetime.now() - self.tracked_time_start
            self.timings.append(f"{message if message else "Elapsed time"}: {str(elapsed_time)}")
            if reset_timer:
                self.tracked_time_start = datetime.now()
    
    @staticmethod
    def str_property(property):
        if inspect.isfunction(property) or inspect.ismethod(property):
            return property.__name__
        if callable(property):
            return property.__class__.__name__
        return str(property)
    