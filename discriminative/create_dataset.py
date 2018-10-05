import os
import urllib
import shutil
import zipfile
import logging
import argparse

from tqdm import tqdm

from settings import *

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def check_datasets(datasets_path):
    """
        Try to access to the files and checks if datasets are in the data directory
       In case the files are not found try to download them from original location
    :param datasets_path:
            Path with Food Image dataset zip files
    :return: dict
    """

    datasets_path_dict = {}
    for name in [NAME_FOOD_5K, NAME_FOOD_11K]:
        dataset_path = os.path.join(datasets_path, name)
        # Check whether the dir of the dataset exists
        logger.info('Checking directory {0}'.format(dataset_path))
        if os.path.exists(dataset_path):
            datasets_path_dict[name] = dataset_path
            continue

        logger.info('... directory {0} not found'.format(dataset_path))
        # When the dir of the dataset does not exist, look for the zip file
        dataset_zip_filename = '.'.join([name, 'zip'])
        dataset_zip_path = os.path.join(datasets_path, dataset_zip_filename)
        logger.info('Checking zip file {0} in path {1}'.format(dataset_zip_path, datasets_path))

        if not os.path.exists(dataset_zip_path):
            # When dataset not found, try to download it from origin
            logger.info('... dataset zip file {0} not found'.format(dataset_zip_filename))
            origin = ''.join([ORIGIN, dataset_zip_filename])
            if not os.path.exists(datasets_path):
                logger.info('... creating directory {0}'.format(datasets_path))
                os.makedirs(datasets_path)
            logger.info('... downloading data from {0}'.format(origin))
            urllib.urlretrieve(url=origin, filename=dataset_zip_filename)

        zip_ref = zipfile.ZipFile(dataset_zip_path, 'r')
        logger.info('... extracting data from {0}'.format(dataset_zip_path))
        zip_ref.extractall(dataset_path)
        zip_ref.close()

    return datasets_path_dict


def create_dataset(datasets_path_dict, target_path):
    """
        Copy the images files to the new path, adding a new non food class
    :param datasets_path_dict:
    :param target_path:
    :return:
    """

    def get_class_filename(image_filename):
        return image_filename.split("_")[0]

    food_5k_path = datasets_path_dict[NAME_FOOD_5K]
    food_11k_path = datasets_path_dict[NAME_FOOD_11K]
    logger.info('Creating dataset in path {0}'.format(target_path))
    for dir_name in ['training/', 'validation/', 'evaluation/']:
        food5k_path_dir = os.path.join(food_5k_path, dir_name)
        food_11k_path_dir = os.path.join(food_11k_path, dir_name)
        target_path_dir = os.path.join(target_path, dir_name)
        if not os.path.exists(target_path_dir):
            logger.info('Creating directory {0}'.format(target_path_dir))
            os.makedirs(target_path_dir)
        food_5k_filename_path_list = [os.path.join(food5k_path_dir, f) for f in os.listdir(food5k_path_dir)]
        food_11k_filename_path_list = [os.path.join(food_11k_path_dir, f) for f in os.listdir(food_11k_path_dir)]

        # Copying food images to new dataset
        logger.info("Copying {} files from {}".format(len(food_11k_filename_path_list), food_11k_path_dir))
        for filename_path in tqdm(food_11k_filename_path_list):
            _, filename = os.path.split(filename_path)  # Separate base from extension
            filename_path_dir = os.path.join(target_path_dir, get_class_filename(filename))
            if not os.path.exists(filename_path_dir):
                os.makedirs(filename_path_dir)
            new_filename = os.path.join(filename_path_dir, filename)
            if not os.path.exists(new_filename):
                shutil.copy(filename_path, new_filename)

        # Copying non food images to new dataset
        logger.info("Copying {} files from {}".format(len(food_5k_filename_path_list), food5k_path_dir))
        for filename_path in tqdm(food_5k_filename_path_list):
            _, filename = os.path.split(filename_path)  # Separate base from extension
            base, extension = os.path.splitext(filename)
            class_id, image_id = base.split("_")
            if class_id != FOOD_5K_ORIGIN_CLASSID:
                continue
            new_filename_base = ''.join(['_'.join([FOOD_11K_TARGET_CLASSID, image_id]), extension])
            filename_path_dir = os.path.join(target_path_dir, FOOD_11K_TARGET_CLASSID)
            if not os.path.exists(filename_path_dir):
                os.makedirs(filename_path_dir)
            new_filename = os.path.join(filename_path_dir, new_filename_base)
            if not os.path.exists(new_filename):
                shutil.copy(filename_path, new_filename)


def get_args():
    """
        This method parses and return arguments passed in
    :return:
    """
    parser = argparse.ArgumentParser(
        description='Create FOOD_12 dataset')
    # Add arguments
    parser.add_argument(
        '-s', '--source', type=str, help='Path to the Food Image directories or zip files', required=True)
    parser.add_argument(
        '-t', '--target', type=str, help='Target path for the dataset', required=False)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    source_path = args.source
    target_path = args.target
    if target_path is None:
        target_path = os.path.join(source_path, 'Food12')
    # Return all variable values
    return source_path, target_path


def run():
    source_path, target_path = get_args()
    datasets_path_dict = check_datasets(source_path)
    create_dataset(datasets_path_dict, target_path)


if __name__ == '__main__':
    run()
