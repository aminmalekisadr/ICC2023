import sys
import time
import logging
import yaml
import pdb
from preprocessing import Preprocessor
def main():
    """Main block of code, which runs the data-preprocessing"""
    start = time.time()
    logging.basicConfig(level="INFO")

    # Load the .yaml data
    assert len(sys.argv) == 2, "Exactly one experiment configuration file must be "\
        "passed as a positional argument to this script. \n\n"\
        "E.g. `python preprocess_dataset.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        logging.info("Loading simulation settings from %s", sys.argv[1])
        experiment_config = yaml.safe_load(yaml_config_file)
    data_parameters = experiment_config['data_parameters']
    ga_parameters = experiment_config['ga_parameters']
    data_preprocessor = Preprocessor(data_parameters)
    for name in data_preprocessor.SMAP:
        try:
           scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y= data_preprocessor.load_dataset(name)




        except:
           print("I read all the files")

    pdb.set_trace()

    # Prints elapsed time
    end = time.time()
    exec_time = (end-start)
    logging.info("Total execution time: %.5E seconds", exec_time)

















if __name__ == '__main__':
    main()
