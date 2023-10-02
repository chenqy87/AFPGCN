import configparser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", default=r'./experiment.conf',
                    type=str, help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config_path)
data_config = config["Data"]
model_config = config["Model"]
test_config = config["Test"]

NODES = int(data_config["nodes"])
TRAINING_PERCENT = float(data_config["training_percent"])
VALIDATION_PERCENT = float(data_config["validation_percent"])
SUPPORT_PERCENT = float(data_config["support_percent"])
POINTS_PER_HOUR = int(data_config["points_per_hour"])

LEARNING_RATE = float(model_config["LEARNING_RATE"])
META_LEARNING_RATE = float(model_config["META_LEARNING_RATE"])
REP_LEARNING_RATE = float(model_config["REP_LEARNING_RATE"])
NUM_FOR_PREDICT = int(model_config["NUM_FOR_PREDICT"])
LEN_INPUT = int(model_config["LEN_INPUT"])
IN_CHANNELS = int(model_config["IN_CHANNELS"])
NB_BLOCK = int(model_config["NB_BLOCK"])
K = int(model_config["K"])
NB_CHEV_FILTER = int(model_config["NB_CHEV_FILTER"])
NB_TIME_FILTER = int(model_config["NB_TIME_FILTER"])
BATCH_SIZE = int(model_config["BATCH_SIZE"])
NUM_OF_WEEKS = int(model_config["NUM_OF_WEEKS"])
NUM_OF_DAYS = int(model_config["NUM_OF_DAYS"])
NUM_OF_HOURS = int(model_config["NUM_OF_HOURS"])
TIME_STRIDES = int(model_config["TIME_STRIDES"])
LOSS_FUNCTION = model_config["LOSS_FUNCTION"]
METRIC_METHOD = model_config["METRIC_METHOD"]
MISSING_VALUE = float(model_config["MISSING_VALUE"])
TEST_UPDATE_STEP = int(model_config["TEST_UPDATE_STEP"])
REPTILE_INNER_STEP = int(model_config["reptile_inner_step"])
MODEL_SIZE = int(model_config["model_size"])

RESULT_SAVE_NUMBER_START = int(test_config["result_save_number_start"])
RESULT_SAVE_NUMBER_END = int(test_config["result_save_number_end"])
TEST_FREQUENCY = int(test_config["test_frequency"])
TRAINING_EPOCHS = int(test_config["training_epochs"])
BATCH_INDEX = int(test_config["batch_index"])
