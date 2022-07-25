from conf import *
from train import *

def main(config):
    train(config)
    
if __name__ == '__main__':
    config = parse_arg()
    main(vars(config))