from constants import LOG_DIR
import os
import datetime


def log_setup(log_dir=LOG_DIR):
	now = datetime.datetime.now().strftime("%b:%d:%Y:%H:%M:%S")
	folder_name = log_dir + now + "/"
	os.mkdir(folder_name)
	return folder_name


class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__


def merge(list_of_lists) :
    return [sublist[i] for sublist in list_of_lists
            for i in range(len(sublist)) ]