from optparse import OptionParser


def build_parser():
    parser = OptionParser()
    parser.add_option(
        '--app-directory',
        dest = 'app_directory',
        default = 'birds-eye-view-predictor',
        type = 'string',
        help = 'base location for the project'
    )

    return parser