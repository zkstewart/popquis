# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.locations import Locations

def validate_args(args):
    # Validate numeric arguments
    if args.threads < 1:
        raise ValueError("-t must be a positive integer")
    if args.cmMbp <= 0:
        raise ValueError("--centimorgans must greater than zero")
    if args.weakDistance < 1:
        raise ValueError("--weak must be a positive integer")
    if args.moderateDistance < 1:
        raise ValueError("--moderate must be a positive integer")
    if args.strongDistance < 1:
        raise ValueError("--strong must be a positive integer")
    
    # Stitch combination nargs into a single string
    args.combination = " ".join(args.combination)
    
    # Create locations object
    args.locations = Locations(args.outputDirectory) # internally validates args.outputDirectory
