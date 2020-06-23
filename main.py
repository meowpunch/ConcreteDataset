import sys

from core import ConcretePipeline
from utils.logger import init_logger

logger = init_logger()


def main(p_type):
    """
    :return: exit code
    """
    try:
        ConcretePipeline(p_type=p_type).process()
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
                No Feature Extraction
            'processed':
                Linear - ElasticNet
                + Feature Extraction
            'advanced':
                Ensemble - GradientBoosting
                + Feature Extraction
                
        ex) shell script (cmd on windows 10)
            > python main.py advanced
    """
    try:
        process_type = sys.argv[1]
        main(p_type=process_type)
    except IndexError:
        logger.error(msg="please pass process_type as argument", exc_info=False)
