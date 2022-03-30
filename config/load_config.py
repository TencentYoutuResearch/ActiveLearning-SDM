import configparser
import os
import sys
sys.path.append('..')
from util.utils import completion_name

# config abspath
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load config
config = configparser.ConfigParser()
config.read(os.path.join(BASE_DIR,'ini.config'),encoding='utf-8')

# active settings
SAMPLE_STRATEGY = config.get('sample','strategy')
NUM_INIT_LB = int(config.get('sample','init'))
QUERY_RATIO = float(config.get('sample','query_ratio'))
NUM_TOTAL = int(config.get('sample','totality'))

# data settings
DATA_NAME = config.get('data','name')
DATA_PATH = config.get('data','path')
DATA_SOURCE = completion_name(config.get('data','source'), DATA_NAME)
DATA_TARGET = completion_name(config.get('data','target'), DATA_NAME)
DATA_CLASS = int(config.get('data','class'))

# learning process settings
BATCH_SIZE = int(config.get('param','batch'))
LEARN_RATE = float(config.get('param','lr'))
EPOCH = int(config.get('param','epoch'))
WEIGHT_DECAY = float(config.get('param','weight_decay'))
DEVICE = config.get('param','device')
USE_CUDA = bool(config.get('param','use_cuda'))
NUM_WORK = int(config.get('param','num_worker'))
SEED = int(config.get('param','seed'))
LOG_INTERVAL = int(config.get('param','log_interval'))
N_VIEWS = int(config.get('param','n_views'))
EMA_MOMENTUM = float(config.get('param','ema_momentum'))
SDM_LAMBDA = float(config.get('param','sdm_lambda'))
SDM_MARGIN = float(config.get('param','sdm_margin'))

# others
REVERSE_WEIGHT = float(config.get('mme','reverse_weight'))
MME_LAMBDA = float(config.get('mme','lambda'))
