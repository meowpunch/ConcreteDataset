import sys


def selector(key):
    return {
        'baseline': None,
        'advanced': None
    }[key]


def main(p_type):
    """
    :return: exit code
    """
    try:
        selector(key=p_type)
    except KeyError:
        print("please pass proper arguments, such as 'baseline'")
        return False

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
        print("please pass process_type as argument")
