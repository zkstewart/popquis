# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.experiment import Critic
from modules.spreadsheet import Spreadsheet

IMG_WIDTH = 525
IMG_HEIGHT = 525
IMG_DPI = 150
LABEL_WIDTH = IMG_WIDTH//4

def write_report_tsv(locations, configuration):
    with open(locations.outputTSV, "w") as fileOut:
        fileOut.write(f"pop_balance\tphenotype_error\tqtl\tpop_size\tno_signal\tweak_signal\tmoderate_signal\tstrong_signal\n")
        
        for (popBalance, phenotypeError), popSizes in configuration:
            if popSizes is None:
                continue
            
            # Obtain the Spreadsheet this configuration will have results stored within
            spreadsheet = Spreadsheet(locations.storageDir, popBalance, phenotypeError, popSizes)
            spreadsheet.load()
            
            for i, strengths in enumerate(spreadsheet.get_strengths()):
                for y, size in enumerate(popSizes):
                    no, weak, moderate, strong = strengths[y]
                    fileOut.write(f"{popBalance}\t{phenotypeError}\t{i+1}\t{size}\t{no}\t{weak}\t{moderate}\t{strong}\n")

def _blank_plot_segment():
    try:
        plt.close(1)
    except:
        pass
    
    fig = plt.figure(figsize=(IMG_WIDTH/IMG_DPI, IMG_HEIGHT/IMG_DPI), dpi=IMG_DPI,
                     num=1, clear=True) # prevent memory leak
    ax = fig.add_subplot()
    
    ax.text(0.5, 0.5, "N/A",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes)
    
    ax.axis("off")
    
    # Convert to PIL Image object
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis = 2)
    
    plotImg = Image.frombytes("RGBA", (w , h), buf.tobytes())
    return plotImg

def _popsize_label_segment(popSize):
    try:
        plt.close(1)
    except:
        pass
    
    fig = plt.figure(figsize=(LABEL_WIDTH/IMG_DPI, IMG_HEIGHT/IMG_DPI), dpi=IMG_DPI,
                     num=1, clear=True) # prevent memory leak
    ax = fig.add_subplot()
    
    ax.text(0.5, 0.5, popSize,
        horizontalalignment="center", 
        verticalalignment="center", 
        transform=ax.transAxes)
    
    ax.axis("off")
    
    # Convert to PIL Image object
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis = 2)
    
    plotImg = Image.frombytes("RGBA", (w , h), buf.tobytes())
    return plotImg

def _qtl_ed_plot(y, score):
    '''
    Obtains a plot (as a Pillow Image object) for visualising the line fit to the ED^4 data.
    
    Credit to https://www.icare.univ-lille.fr/how-to-convert-a-matplotlib-figure-to-a-numpy-array-or-a-pil-image/
    '''
    try:
        plt.close(1)
    except:
        pass
    
    # Plot the image with line fit
    fig = plt.figure(figsize=(IMG_WIDTH/IMG_DPI, IMG_HEIGHT/IMG_DPI), dpi=IMG_DPI,
                     num=1, clear=True, layout="tight") # prevent memory leak
    ax = fig.add_subplot()
    
    ax.plot(np.arange(0, len(y)), y)
    #ax.scatter(x, y, label="SNP segregation")
    ax.set_xlabel("Variant number")
    ax.set_ylabel("$ED^4$")
    ax.set_title(round(score, 4))
    
    # Convert to PIL Image object
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis = 2)
    
    plotImg = Image.frombytes("RGBA", (w , h), buf.tobytes())
    return plotImg

def plot_replicate_exemplars(locations, configuration):
    blank = _blank_plot_segment()
    
    for (popBalance, phenotypeError), popSizes in configuration:
        if popSizes is None:
            continue
        
        # Obtain the Spreadsheet this configuration will have results stored within
        spreadsheet = Spreadsheet(locations.storageDir, popBalance, phenotypeError, popSizes)
        spreadsheet.load()
        
        # Iterate over each simulated QTL
        for i, (qtlED, qtlScores) in enumerate(zip(spreadsheet.get_ed(), spreadsheet.get_scores())):
            plotFileName = os.path.join(locations.qcPlotsDir,
                                        f"{popBalance}_{phenotypeError}.qtl{i+1}.png")
            if os.path.isfile(plotFileName):
                continue
            
            # Format an Image for later output
            nrow = len(popSizes)
            ncol = 4 # one exemplar for strengths: none, weak, moderate, strong
            image = Image.new("RGB", (LABEL_WIDTH + IMG_WIDTH*ncol, IMG_HEIGHT*nrow))
            y_offset = -IMG_HEIGHT
            for popSizeED, popSizeScore, popSize in zip(qtlED, qtlScores, popSizes):
                y_offset += IMG_HEIGHT
                
                exemplars = [None, None, None, None]
                remainingExemplars = len(exemplars)
                for replicateED, replicateScore in zip(popSizeED, popSizeScore):
                    thisStrength = Critic.scores_to_strength(np.array([[replicateScore]]))[0]
                    exemplarIndex = 0 if thisStrength[0] else 1 if thisStrength[1] else 2 if thisStrength[2] else 3
                    if exemplars[exemplarIndex] is None:
                        exemplars[exemplarIndex] = _qtl_ed_plot(replicateED, replicateScore)
                        remainingExemplars -= 1
                    if remainingExemplars == 0:
                        break
                
                # Store the exemplar plots
                labelPlot = _popsize_label_segment(popSize)
                
                x_offset = -IMG_WIDTH
                for i, exemplarPlot in enumerate(exemplars):
                    x_offset += IMG_WIDTH
                    
                    if i == 0:
                        image.paste(labelPlot, (x_offset, y_offset))
                        x_offset += LABEL_WIDTH
                    if exemplarPlot is None:
                        image.paste(blank, (x_offset, y_offset))
                    else:
                        image.paste(exemplarPlot, (x_offset, y_offset))
            
            image.save(plotFileName)
