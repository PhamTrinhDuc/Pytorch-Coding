import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import glob
import os
import shutil
import argparse
import numpy as np
from typing import List
from utils.app_path import AppPath
from utils.logger import Logger
from config.catdog_config import CatDogArgs


LOGGER = Logger(name=__file__, log_file="log process data.log")
LOGGER.log.info("Starting processing data")

def create_train_data(version: str, 
                      source_dir: List[str],
                      dest_dir: str,
                      ratio: List[float]):
    
    LOGGER.log.info("Begin create train/val/test data")
    LOGGER.log.info(f"Version: {version}")
    LOGGER.log.info(f"Source dir: {source_dir}")
    LOGGER.log.info(f"Destination dir: {dest_dir}")
    LOGGER.log.info(f"Ratio [train, val]: {ratio}")

    for label in CatDogArgs.classes:
        os.makedirs(f"{dest_dir}/{version}/train/{label}", exist_ok=True)
        os.makedirs(f"{dest_dir}/{version}/val/{label}", exist_ok=True)
        os.makedirs(f"{dest_dir}/{version}/test/{label}", exist_ok=True)
    
    for label in CatDogArgs.classes:
        LOGGER.log.info(f"Processing {label} ...")
        all_label_files = []
        for source in source_dir:
            all_label_files.extend(glob.glob(f"{source/label}/*.jpg"))
        
        num_label_files = len(all_label_files)

        LOGGER.log.info(f"Number of files the {label}: {num_label_files}")

        all_label_files = np.array(all_label_files)
        shuffle_indices = np.random.permutation(num_label_files)

        num_train = int(ratio[0] * num_label_files)
        num_val = int(ratio[1] * num_label_files)
        # num_test = num_label_files - num_train - num_test

        train_files = all_label_files[shuffle_indices][:num_train]
        val_files = all_label_files[shuffle_indices][num_train: num_train + num_val]
        test_files = all_label_files[shuffle_indices][num_train + num_val: ]

        for file in train_files:
            dest_file = f"{dest_dir}/{version}/train/{label}/{os.path.basename(file)}"
            shutil.copy(src=file, dst=dest_file)
        
        for file in val_files:
            dest_file = f"{dest_dir}/{version}/val/{label}/{os.path.basename(file)}"
            shutil.copy(src=file, dst=dest_file)
        
        for file in test_files:
            dest_file = f"{dest_dir}/{version}/test/{label}/{os.path.basename(file)}"
            shutil.copy(src=file, dst=dest_file)

    LOGGER.log.info(f"Finish create train/val/test data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True,
                        help="Version of the data, e.g v1, as name of the folder")
    parser.add_argument("--merge-collected", action="store_true", 
                        help="Merge collected data to raw data")
    parser.add_argument("--dest_dir", type=str, default=AppPath.TRAIN_DATA_DIR, 
                        help="Destination of the directory to save the train/val/test data")
    parser.add_argument("--ratio", type=float, nargs="+", default=[0.6, 0.2], 
                        help="Ratio of the train/val/test data")
    
    args = parser.parse_args()

    if os.path.exists(AppPath.TRAIN_DATA_DIR / args.version):
        shutil.rmtree(AppPath.TRAIN_DATA_DIR / args.version)
    

    # merge root data and collected data to root data
    source_dir = [AppPath.RAW_DATA_DIR]
    if args.merge_collected:
        source_dir += [AppPath.COLLECTED_DATA_DIR]

    create_train_data(version=args.version, 
                      source_dir=source_dir, 
                      dest_dir=args.dest_dir, 
                      ratio=args.ratio)
    