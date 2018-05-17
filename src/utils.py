


def compute_log_dir(log_dir):
	now = datetime.datetime.now().strftime("%b:%d:%Y:%H:%M:%S")
	return log_dir + "run_logs:" + now + "/"