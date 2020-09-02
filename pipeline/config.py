import logging
import sys

config = {
    'ldc_kg_dir': '../pipeline_test/ldc',
    # 'wd_kg_dir': '../pipeline_test/wd',
    'wd_to_fb_file': '../pipeline_test/df_wd_fb_20200803.csv',
    'input_dir': '../pipeline_test/input',
    'output_dir': '../pipeline_test/output',
    'run_name': '',  # this can be empty string, which indicates there's no run-name subdirectory
    'temp_dir': '../pipeline_test/temp',
    'namespace_file': '../pipeline_test/aida-namespaces.tsv',
    'logging_level': logging.INFO,
    'num_of_processor': 1
}


def get_logger(name):
    logger = logging.getLogger('gaia-ta2-{}'.format(name))
    logger.setLevel(config['logging_level'])
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    return logger

# params can be overwritten by external config files
