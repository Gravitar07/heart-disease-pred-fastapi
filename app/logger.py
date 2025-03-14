import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from app.config import BASE_DIR

# Define base directory and create logs directory
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Create separate log files for different log levels
log_file = os.path.join(LOG_DIR, f'app_{datetime.now().strftime("%Y%m%d")}.log')
error_log_file = os.path.join(LOG_DIR, f'error_{datetime.now().strftime("%Y%m%d")}.log')

# Define log formatters
standard_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
)
error_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s\n'
    'Exception:\n%(exc_info)s\n'
)

# Create and configure handlers
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10485760,  # 10MB
    backupCount=10
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(standard_formatter)

error_file_handler = RotatingFileHandler(
    error_log_file,
    maxBytes=10485760,  # 10MB
    backupCount=10
)
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(error_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(standard_formatter)

# Configure root logger
logging.basicConfig(level=logging.INFO, handlers=[])
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.addHandler(error_file_handler)
root_logger.addHandler(console_handler)

# Create application logger
logger = logging.getLogger('heart_disease_prediction_app')
logger.setLevel(logging.INFO) 