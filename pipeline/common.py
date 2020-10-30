import subprocess


def exec_sh(s, logger):
    logger.debug('exec_sh:' + s)
    process = subprocess.Popen(s, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0 or stderr != b'':
        logger.error('exec_sh: %s . return code: %s . stderr: %s', s, process.returncode, stderr)
    return stdout, stderr