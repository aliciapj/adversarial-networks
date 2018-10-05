
########################################################################################################################
# Dataset settings
########################################################################################################################

ORIGIN = 'http://grebvm2.epfl.ch/lin/food/'
# Food-5K dataset
# This is a dataset containing 2500 food and 2500 non-food images, for the task of food/non-food classification
# The dataset is divided in three parts: training, validation and evaluation.
# The naming convention is as follows: {ClassID}_{ImageID}.jpg
#    ClassID: 0 or 1; 0 means non-food and 1 means food.
#    ImageID: ID of the image within the class.
NAME_FOOD_5K = "Food-5K"
FOOD_5K_ORIGIN_CLASSID = '0'

# Food-11K dataset
# This is a dataset containing 16643 food images grouped in 11 major food categories.
# The 11 categories are Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup,
# and Vegetable/Fruit.
# Similar as Food-5K dataset, the dataset is divided in three parts: training, validation and evaluation.
# The same naming convention is used, where ID 0-10 refers to the 11 food categories respectively.
# This is a dataset containing 2500 food and 2500 non-food images, for the task of food/non-food classification
NAME_FOOD_11K = "Food-11"
FOOD_11K_TARGET_CLASSID = '11'  # 11 is the 12th category, the one for non food

########################################################################################################################
# Model settings
########################################################################################################################

BATCH_SIZE = 32
OUTPUT_MODEL_FILE = "inceptionv3-ft.model"

IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3
NB_EPOCHS_FINETUNE = 100
NB_EPOCHS_TRANSFERLEARNING = 10

BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 120
