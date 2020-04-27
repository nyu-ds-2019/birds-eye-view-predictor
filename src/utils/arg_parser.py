from optparse import OptionParser


def build_parser():
    parser = OptionParser()
    parser.add_option(
        '--app-directory',
        dest = 'app_directory',
        default = '.',
        type = 'string',
        help = 'base location for the project'
    )

    parser.add_option(
        '--batch-size',
        dest = 'batch_size',
        default = 64,
        type = 'int',
        help = 'batch size to process data'
    )

    parser.add_option(
        '--num-workers',
        dest = 'num_workers',
        default = 2,
        type = 'int',
        help = 'GPU workers'
    )

    parser.add_option(
        '--num-gpus',
        dest = 'num_gpus',
        default = 1,
        type = 'int',
        help = 'number of GPUs'
    )

    parser.add_option(
        '--learning-rate',
        dest = 'learning_rate',
        default = 1e-3,
        type = 'float',
        help = 'learning rate'
    )

    parser.add_option(
        '--num-epochs',
        dest = 'num_epochs',
        default = 50,
        type = 'int',
        help = 'number of epochs'
    )

    parser.add_option(
        '--random-seed',
        dest = 'random_seed',
        default = 0,
        type = 'int',
        help = 'random seed'
    )


    return parser