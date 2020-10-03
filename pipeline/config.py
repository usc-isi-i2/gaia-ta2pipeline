import logging
import sys
import os


def get_env_var(name, optional=False, default=None):
    v = os.environ.get(name)
    if not v:
        if not optional:
            raise Exception('{name} is not properly set'.format(name=name))
        return default
    return v


prod_mode = True if get_env_var('PROD', optional=True) else False


def get_config():
    return \
        {  # development
            'ldc_kg_dir': '../pipeline_test/ldc2019',
            # 'wd_kg_dir': '../pipeline_test/wd',
            'wd_to_fb_file': '../pipeline_test/df_wd_fb_20200803.csv',
            'input_dir': '../pipeline_test/input',
            'output_dir': '../pipeline_test/output',
            'run_name': 'bbn',
            'temp_dir': '../pipeline_test/temp',
            'namespace_file': '../pipeline_test/aida-namespaces-bbn.tsv',
            'logging_level': logging.INFO,
            'num_of_processor': 1,
            'kb_to_fbid_mapping':  None, #'../pipeline_test/kb_to_wd_mapping.json',
        } if not prod_mode else \
        {  # production
            'ldc_kg_dir': os.path.join(get_env_var('REPO_KB'), 'data'),
            'wd_to_fb_file': os.path.join(get_env_var('WD_FB_MAPPING')),
            'input_dir': get_env_var('INPUT'),
            'output_dir': get_env_var('OUTPUT'),
            'run_name': get_env_var('RUN_NAME'),
            'temp_dir': get_env_var('TEMP', optional=True, default='/tmp'),
            'namespace_file': os.path.join(get_env_var('NAMESPACE')),
            'logging_level': logging.INFO,
            'num_of_processor': int(get_env_var('NUM_PROC', optional=True, default='2')),
            'kb_to_fbid_mapping': get_env_var('KB_FBID_MAPPING', optional=True),
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
