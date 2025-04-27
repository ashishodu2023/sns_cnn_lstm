#!/usr/bin/env python
# main.py

import argparse
import logging
from training.train import train_workflow
from testing.test import test_workflow

def setup_logging():
    logging.basicConfig(
        filename='logs/app.log',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly Detector Workflow")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Run training workflow")
    group.add_argument("--test", action="store_true", help="Run testing workflow")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logging()

    if args.train:
        logger.info("Train mode selected.")
        train_workflow(logger)
    elif args.test:
        logger.info("Test mode selected.")
        test_workflow(logger)

if __name__ == "__main__":
    main()