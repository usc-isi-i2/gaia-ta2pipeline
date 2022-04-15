import logging
import sys
import os


DEBUG_LEVEL = {
    # 'CRITICAL': logging.CRITICAL,
    # 'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
}


def get_env_var(name, optional=False, default=None):
    v = os.environ.get(name)
    if not v:
        if not optional:
            raise Exception('{name} is not properly set'.format(name=name))
        return default
    return v


prod_mode = get_env_var('PROD', optional=True, default='False').lower() == 'true'


def get_config():
    return \
        {  # development
            'input_dir': '../pipeline2_test/input',
            'output_dir': '../pipeline2_test/output',
            'run_name': 'uiuc',
            'subrun_name': 'NIST',
            'temp_dir': '../pipeline2_test/temp',

            'logging_level': logging.INFO,
            'num_of_processor': 1,
            'extract_mention': True,

            'namespace_file': '../pipeline2_test/res/aida-namespaces-base.tsv',
            'kgtk_labels': '../pipeline2_test/res/labels.en.100.tsv.gz',
            'kgtk_p279': '../pipeline2_test/res/derived.P279star.tsv.gz',
            # 'kgtk_search_url': 'https://kgtk.isi.edu/api',
            # 'kgtk_similarity_url': 'https://kgtk.isi.edu/similarity_api',
            # 'kgtk_similarity_chunk_size': 20,
            # 'kgtk_nearest_neighbor_url': 'https://kgtk.isi.edu/nearest-neighbors',
            # 'kgtk_nearest_neighbor_k': 100,
            # 'kgtk_es_url': 'http://ckg07.isi.edu:9200',
            # 'kgtk_es_index': 'wikidatadwd-augmented',
        } if not prod_mode else \
        {  # production
            'input_dir': get_env_var('INPUT'),
            'output_dir': get_env_var('OUTPUT'),
            'run_name': get_env_var('RUN_NAME'),
            'subrun_name': get_env_var('SUBRUN_NAME'),
            'temp_dir': get_env_var('TEMP', optional=True, default='/tmp'),

            'logging_level': DEBUG_LEVEL.get(get_env_var('LOGGING', optional=True, default='INFO'), logging.INFO),
            'num_of_processor': int(get_env_var('NUM_PROC', optional=True, default='2')),
            'extract_mention': get_env_var('EXTRACT_MENTION', optional=True, default='False').lower() == 'true',

            'namespace_file': os.path.join(get_env_var('NAMESPACE')),
            'kgtk_labels': get_env_var('KGTK_LABELS'),
            'kgtk_p279': get_env_var('KGTK_P279'),
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
