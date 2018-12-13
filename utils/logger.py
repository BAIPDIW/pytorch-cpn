import logging
from pathlib import Path
import time
def create_logger(log_dir):
    log_dir = Path(log_dir)
    if not log_dir.exists():
        print('=> creating {}'.format(log_dir))
        log_dir.mkdir()
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)

    final_log_file = log_dir/log_file

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    
    return logger