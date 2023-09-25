import pandas as pd
import numpy as np
import csv
import argparse
import logging
from certify import fin_certify
from compute_params import auto_params


def create_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser

if __name__ == "__main__":
    parser = create_arg_parse()
    args = parser.parse_args()
    logger = logging.getLogger("__auto_certify__")

    radius = 3
    search_exponent =50

    if not args.verbose:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    with open("output_certify.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "center_answer", "ball_radius", "certified_radius", "probability"])

        with open("smooth.csv", "r") as g:
            reader_smooth = csv.reader(g)
            row1 = next(reader_smooth)
            for i, row in enumerate(reader_smooth):
                question, center= row[0], row[1]
                params = auto_params(question, radius)
                radius = fin_certify(i, "question_prompt"+str(i)+"_3", center, radius, len(question.split()), params["k"], 
                            params["alpha"], int(params["delta"]*100), search_exponent, params["alpha_2"], params["m"])
                writer.writerow([question, center, radius, args.radius, str(1-params["alpha_2"]-params["alpha_1"])])
            g.close()
        f.close()
