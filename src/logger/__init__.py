import logging
import os

def get_logger(script_name: str, logfile_name: str):
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok = True)

    logger = logging.getLogger(script_name)
    logger.setLevel('DEBUG')

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel('DEBUG')

    file_path = os.path.join(log_dir, logfile_name)
    fileHandler = logging.FileHandler(file_path)
    fileHandler.setLevel('DEBUG')

    Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fileHandler.setFormatter(Formatter)
    consoleHandler.setFormatter(Formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    
    return logger