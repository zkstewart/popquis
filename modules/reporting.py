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

STACK_COLOURS = ["#E24A33", "#348ABD", "#988ED5", "#777777"]

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

def plot_report(locations, configuration):
    SIGNAL_COLUMNS = ["no_signal", "weak_signal", "moderate_signal", "strong_signal"]
    NUM_X_TICKS = 8
    
    if not (os.path.isfile(locations.outputTSV) and os.path.isfile(args.locations.outputTSV + Locations.OKAY_SUFFIX)):
        raise FileNotFoundError(f"Results report file '{locations.outputTSV}' must exist with an associated " + 
                                f"'{Locations.OKAY_SUFFIX}' flag file for a plot to be produced.")
    
    # Parse the tabulated data and derive some global parameters
    reportDF = pd.read_csv(locations.outputTSV, sep="\t")
    bootstraps = reportDF.iloc[0][SIGNAL_COLUMNS].sum()
    minPopSize = reportDF["pop_size"].min()
    maxPopSize = reportDF["pop_size"].max()
    
    # Format a composite plot
    for qtlNum in reportDF["qtl"].unique():
        # Derive file names and skip if they exist
        outputPNG = locations.outputPNG(qtlNum)
        outputPDF = locations.outputPDF(qtlNum)
        
        if os.path.isfile(outputPNG) and os.path.isfile(outputPDF):
            print(f"# Stacked barplots for QTL #{qtlNum} already exist; skipping...")
            continue
        
        # Setup figure object
        ncol = int(len(configuration.phenotypeError) / 2) # 2==faceted rows; should result in 3
        nrow = int(len(configuration.popBalance) * 2) + 1 # +1 is a spacing row
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol,
                                 figsize=(20, 8),
                                 dpi=300,
                                 layout="constrained")
        
        # Blank the middle spacing row
        for colIndex in range(ncol):
            axes[len(configuration.popBalance), colIndex].axis("off")
        
        # Iterate over each configuration combination
        for i, phenotypeError in enumerate(configuration.phenotypeError):
            colIndex = i % ncol
            
            for x, popBalance in enumerate(configuration.popBalance[::-1]): # largest to smallest
                rowIndex = x + ( (i // ncol) * len(configuration.popBalance))
                if rowIndex >= len(configuration.popBalance): # skip the spacing row
                    rowIndex += 1
                ax = axes[rowIndex, colIndex]
                
                # Obtain data
                plotDF = reportDF[
                    (reportDF["phenotype_error"] == phenotypeError) & \
                    (reportDF["pop_balance"] == popBalance) & \
                    (reportDF["qtl"] == qtlNum)
                ]
                
                # Calculate signal threshold values
                milestones = Calculator.interpolate_popsize_milestones(plotDF, bootstraps)
                
                # Create stackplot of the data
                stack = ax.stackplot(plotDF["pop_size"],
                            (
                                plotDF["strong_signal"],
                                plotDF["moderate_signal"],
                                plotDF["weak_signal"],
                                plotDF["no_signal"]
                            ),
                            labels=["Strong", "Moderate", "Weak", "None"],
                            colors=STACK_COLOURS)
                ax.margins(0,0) # fill all of the rectangular space
                
                # Mark the 95% cutoffs
                if milestones["atleast_weak"]:
                    mark1 = ax.vlines(milestones["atleast_weak"],
                            ymin=0, ymax=bootstraps,
                            linestyle="--", color="white",
                            label="Weak")
                if milestones["atleast_mid"]:
                    mark2 = ax.vlines(milestones["atleast_mid"],
                            ymin=0, ymax=bootstraps,
                            linestyle="--", color="black",
                            label="Moderate")
                if milestones["strong_signal"]:
                    mark3 = ax.vlines(milestones["strong_signal"],
                            ymin=0, ymax=bootstraps,
                            linestyle="--", color="#FFFF00",
                            label="Strong")
                
                # Set axis limits
                ax.set_ylim(0, bootstraps)
                ax.set_xlim(minPopSize, maxPopSize)
                
                # Set axis aesthetic
                ax.set_xticks(np.asarray(
                    np.linspace(minPopSize, maxPopSize, num=NUM_X_TICKS),
                dtype=np.int64))
                
                ax.set_yticks([0, bootstraps//2, bootstraps],
                              labels=["0%", "50%", "100%"])
                
                # Turn off axis labels if applicable
                if x != len(configuration.popBalance) - 1:
                    ax.set_xticklabels([])
                if colIndex != 0:
                    ax.set_yticklabels([])
        
        # Configure the legend
        legend1 = plt.legend(handles=stack,
                             bbox_to_anchor=(1.05, 1.0),
                             loc="upper right")
        fig.add_artist(legend1)
        
        legend2 = plt.legend(handles=[mark1, mark2, mark3],
                             bbox_to_anchor=(1.05, 1.0),
                             loc="lower right")
        
        # Write output files
        fig.savefig(outputPNG) # pad_inches=1.5
        fig.savefig(outputPDF) # pad_inches=1.5
