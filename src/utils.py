from constants import LOG_DIR
import os
import datetime


def log_setup(log_dir=LOG_DIR):
	now = datetime.datetime.now().strftime("%b:%d:%Y:%H:%M:%S")
	folder_name = log_dir + now + "/"
	os.mkdir(folder_name)
	return folder_name