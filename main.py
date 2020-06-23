import sys

from core import BaselinePipeline
from utils.logger import init_logger

logger = init_logger()


def selector(key):
    return {
        'baseline': BaselinePipeline,
        'advanced': None
    }[key]


def main(p_type):
    """
    :return: exit code
    """
    try:
        pipeline = selector(key=p_type)
        pipeline().process()
    except KeyError:
        logger.error("please pass proper arguments, such as 'baseline'", exc_info=False)

    return True


if __name__ == '__main__':
    """
        - Manual -
        
        python main.py p_type
        
        p_type: 
            'baseline': 
                Linear - ElasticNet
            'advanced':
                
    """
    try:
        process_type = sys.argv[1]
        main(p_type=process_type)
    except IndexError:
        logger.error(msg="please pass process_type as argument", exc_info=False)
