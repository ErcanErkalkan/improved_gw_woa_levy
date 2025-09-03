def get_logger(name='egwwoa'):
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
