import os
import re
import sys

def validate_args(args):
    # Validate numeric arguments
    if args.threads < 1:
        raise ValueError("-t must be a positive integer")
    if args.cmMB <= 0:
        raise ValueError("--centimorgans must greater than zero")
    if args.weakDistance < 1:
        raise ValueError("--weak must be a positive integer")
    if args.moderateDistance < 1:
        raise ValueError("--moderate must be a positive integer")
    if args.strongDistance < 1:
        raise ValueError("--strong must be a positive integer")
    
    # Stitch combination nargs into a single string
    args.combination = " ".join(args.combination)
    
    # Validate output directory
    args.outputDirectory = os.path.abspath(args.outputDirectory)
    if os.path.isdir(args.outputDirectory):
        print(f"# -o location already exists; will attempt to resume a previous run")
    elif not os.path.exists(args.outputDirectory):
        parentDir = os.path.dirname(args.outputDirectory)
        if not os.path.isdir(parentDir):
            raise NotADirectoryError(f"Cannot write to -o '{args.outputDirectory}' as its parent " +
                                     f"directory ({parentDir}) is not a directory or does not exist")
        else:
            os.mkdir(args.outputDirectory)
            print(f"# Created output directory '{args.outputDirectory}' as part of argument validation")
    else:
        raise NotADirectoryError(f"-o location already exists, but is not a directory. Try a different " +
                                 "location instead")
