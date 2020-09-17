import logging
import sys
import os


def get_env_var(name, optional=False):
    v = os.environ.get(name)
    if not v and not optional:
        raise Exception('{name} is not properly set'.format(name=name))
    return v


def get_config():
    return \
        {  # development
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
        } if not get_env_var('PROD', optional=True) else \
        {  # production
            'ldc_kg_dir': os.path.join(get_env_var('REPO_KB'), 'data'),
            # 'wd_kg_dir': '../pipeline_test/wd',
            'wd_to_fb_file': os.path.join(get_env_var('RES'), 'df_wd_fb.csv'),
            'input_dir': get_env_var('INPUT'),
            'output_dir': get_env_var('OUTPUT'),
            'run_name': get_env_var('RUN_NAME', optional=True),
            'temp_dir': get_env_var('TEMP'),
            'namespace_file': os.path.join(get_env_var('RES'), 'aida-namespace.tsv'),
            'logging_level': logging.INFO,
            'num_of_processor': 1
        }


config = get_config()


def get_logger(name):
    logger = logging.getLogger('gaia-ta2-{}'.format(name))
    logger.setLevel(config['logging_level'])
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    return logger

# params can be overwritten by external config files
