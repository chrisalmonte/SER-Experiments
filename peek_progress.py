import pickle
import sys
from cLogger import Log

LOG_PKL_PATH = r""

# In case it wants to be used from the command line
import sys
if len(sys.argv) > 1:
    LOG_PKL_PATH = sys.argv[1]

with open(LOG_PKL_PATH, 'rb') as file:
    log = pickle.load(file)

log.plot_epoch_values(save_path="output/logs/progress_epoch_values.png")
log.path = "output/logs"
log.save_txt()
