import json
import os

import pandas as pd
from loguru import logger


class HrResult(object):
    def __init__(self):
        logger.info("Start to initialize HrResult class.")
        print('a')
        logger.info("HrResult Class initialized.")

    def hr_response(self, result_df, grade):
        logger.info("Formatting output response...")
        whole_result = dict()
        for idx, row in result_df.iterrows():
            pair_dict = dict()
            pair_dict["Question"] = row["QUESTION"]
            pair_dict["Answer"] = row["ANSWER"]
            pair_dict["Image"] = row["IMAGE"]
            pair_dict["SimScore"] = row["SimScore"]
            whole_result[idx] = pair_dict
        # to ensure the result is json format
        # result_json = json.dumps(whole_result)
        return whole_result
