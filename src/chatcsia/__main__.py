import argparse
import sys

from chatcsia._version import __version__

parser = argparse.ArgumentParser(
    prog="chatcsia", description="Chatbot for question answering over document"
)
parser.add_argument(
    "-V", "--version", action="version", version=f"%(prog)s {__version__}"
)


def main(args=None):
    """
    Calls entry point with command line arguments

    accesible with `python -m chatcsia`
    """
    if args is None:
        args = sys.argv[1:]
    parser.parse_args(args)


if __name__ == "__main__":
    sys.exit(main())
