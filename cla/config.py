import json

FILE_LIST = './data/dataset_list_145000_18125_18125_1226.json'
ANNOTATION_PATH = './data/all_data_1226.json'

DATA_PATH = '.'
MODEL_SAVE_PATH = './models/'

DEVICE = 'cuda'

#USE_FEATURE=['tsn', 'slowfast']
USE_FEATURE=['tsn']
FEATURE_DIM = 0
for FEATURE in USE_FEATURE:
    if FEATURE == 'tsn':
        FEATURE_DIM += 2048
    if FEATURE == 'slowfast':
        FEATURE_DIM += 2304 

FEATURE_LEN = 300 # DO NOT CHANGE
TIME_UNIT = 1 # DO NOT CHANGE

GAP = 16 # VALID LOCAL RANGE
CHANNEL_NUM = 4 
ENCODER_HIDDEN = 512 
DECODER_HIDDEN = 128

EVENT_LOSS_COEF = 0.5 
SHOT_LOSS_COEF = 0.2
WHOLE_LOSS_COEF = 0.3

AUX_LOSS_COEF = 0.5   

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DROP_RATE = 0.2

GLUE_PROB = 0.3 # Probability of glueing augmentation 
INTERPOLATION_PROB = 0.2 # Probability of data of DATA_PATH_2

THRESHOLD = 0.1 # Minimum score to be event boundary
SIGMA_LIST = [-1, 0.4] # List of sigma values of gaussian filtering in validation
TEST_THRESHOLD = 0.35
GOAL_SCORE = 0.815 # Train ends when validation score gets here

#PATIENCE = 15 # Patience for early stopping
PATIENCE = 40 # Patience for early stopping

NUM_WORKERS = 0
