# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.spreadsheet import Spreadsheet

def write_report_tsv(locations, configuration, numQTLs):
    with open(locations.outputTSV, "w") as fileOut:
        fileOut.write(f"pop_size\tpop_balance\tphenotype_error\tno_signal\tweak_signal\tmoderate_signal\tstrong_signal\n")
        
        for (popBalance, phenotypeError), popSizes in configuration:
            if popSizes is None:
                continue
            
            # Obtain the Spreadsheet this configuration will have results stored within
            spreadsheet = Spreadsheet(locations.storageDir, popBalance, phenotypeError, popSizes)
            spreadsheet.load()
            
            for x in range(numQTLs):
                strengths = getattr(spreadsheet, f"strengths{x+1}")
                for y, size in enumerate(popSizes):
                    no, weak, moderate, strong = strengths[y]
                    fileOut.write(f"{size}\t{popBalance}\t{phenotypeError}\t{no}\t{weak}\t{moderate}\t{strong}\n")
