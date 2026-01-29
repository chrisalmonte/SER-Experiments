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
        self.epoch_values = {}
        self.timings = []
        self.tracked_time_start = None
        self.last_save = None
    
    def log_message(self, message: str, show: bool=True):
        self.messages.append(f"{datetime.now().strftime('%Y_%m_%d-%H:%M:%S')}: {message}")
        if show:
            print(message)
    
    def log_property(self, key: str, value, show: bool=True):
        property = f"{key}: {self.str_property(value)}"
        self.properties.append(property)
        if show:
            print(property)

    def log_properties(self, set_name:str ,properties: dict, show: bool=True):
        self.properties.append(' ')
        self.properties.append(set_name + ":\n")
        for key, value in properties.items():
            self.properties.append(f"{key}: {self.str_property(value)}")
            if show:
                print(f"{key}: {self.str_property(value)}")
        self.properties.append(' ')

    def log_epoch(self, epoch: int, properties: dict, show=True):
        if show:
            print(f'Epoch {epoch}:' + '\n')
        if 'epoch' not in self.epoch_values:
            self.epoch_values['epoch'] = []
        self.epoch_values['epoch'].append(epoch)

        for key, value in properties.items():
            if key not in self.epoch_values:
                self.epoch_values[key] = []
            self.epoch_values[key].append(value)
            if show:
                print(f"{key}: {value}")

    def save(self, overwrite: bool=True):
        if overwrite and self.last_save:
            file_path = self.last_save
        else:
            file_path = os.path.join(self.path, f"{self.prefix}_{datetime.now().strftime('%Y_%m_%d-%H%M%S')}.pkl")
            self.last_save = file_path
        print(f"Saving log to {file_path}...")
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print("Log saved.")

    def save_txt(self):
        file_path = os.path.join(self.path, f"{self.prefix}_{datetime.now().strftime('%Y_%m_%d-%H%M%S')}.txt")
        with open(file_path, 'w', encoding="utf-8") as file:
            file.write("Log saved on: ")
            file.write(datetime.now().strftime("%b %d %Y - %H:%M:%S") + "\n\n")
            if self.properties:
                file.write("-----Properties-----\n\n")
                file.write("\n".join(self.properties) + "\n\n")
            if self.messages:
                file.write("-----History-----\n\n")
                file.write("\n".join(self.messages) + "\n\n")
            if self.timings:
                file.write("-----Timings-----\n\n")
                file.write("\n".join(self.timings) + "\n\n")
            if self.epoch_values:
                file.write("-----Training-----\n\n")
                for epoch in len(self.epoch_values['epoch']):
                    file.write(f"Epoch {self.epoch_values['epoch'][epoch]}:\n")
                    for property in self.epoch_values.keys():
                        file.write(f"{property}: {self.epoch_values[property][epoch]}\n")
                file.write("\n")                        
        
    def track_time(self, track: bool, message: str="", show: bool=True):
        if not track:
            self.tracked_time_start = None
            self.log_message(message if message else "Time tracking stopped.", show=show)
        else:
            if self.tracked_time_start:
                self.log_message("Time tracking reset.", show=show)
            self.tracked_time_start = datetime.now()
            self.log_message(message if message else "Time tracking started.", show=show)            
    
    def log_elapsed_time(self, message: str="", reset_timer: bool=False, show: bool=True):        
        if not self.tracked_time_start:
            raise ValueError("Time tracking has not been started. Call track_time() first.")
        else:
            elapsed_time = datetime.now() - self.tracked_time_start
            log = f"{message if message else 'Elapsed time'}: {str(elapsed_time)}"
            self.timings.append(log)
            if show:
                print(log)
            if reset_timer:
                self.tracked_time_start = datetime.now()
    
    @staticmethod
    def str_property(property):
        if inspect.isfunction(property) or inspect.ismethod(property):
            return property.__name__
        if callable(property):
            return property.__class__.__name__
        return str(property)
    