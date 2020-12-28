from utils.log_utils import init_logger
from utils.data_utils import save_data
from config.basic_config import configs


if __name__ == "__main__":
    logger = init_logger(
        log_name='utils', log_dir=configs['log_dir'])
    # 将数据保存起来
    save_data(configs['all_data_path'],
              configs['train_data_path'], configs['test_data_path'],
              configs['val_data_path'])
