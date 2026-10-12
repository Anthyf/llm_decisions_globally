#!/usr/bin/env python3
"""
Main script to generate data for LLM Trade-Offs framework.
"""

from utils.utils_base import load_config # remember to change back to utils when country
from tradeoff_data.tradeoff_data import tradeoff_data


def main():
    args = load_config()

    # Run experiments
    tradeoff_data(args)


if __name__ == "__main__":
    main()
