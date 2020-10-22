import subprocess


def exec_sh(s, logger):
    logger.debug('exec_sh:', s)
    process = subprocess.Popen(s, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if stderr != b'':
        logger.error('exec_sh: %s . stderr: %s', s, stderr)
    return stdout, stderr