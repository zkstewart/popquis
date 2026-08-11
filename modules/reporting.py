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
from modules.statistics import Calculator

IMG_WIDTH = 525
IMG_HEIGHT = 525
IMG_DPI = 150
LABEL_WIDTH = IMG_WIDTH//4

STACK_COLOURS = ["#E24A33", "#348ABD", "#988ED5", "#777777"]

def write_raw_tsv(locations, configuration):
    with open(locations.rawTSV, "w") as fileOut:
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

def _separate_thresholds(thresholds, delta, info=None):
    '''
    Modifies the x coordinates of marker threshold lines to ensure
    the visibility of any overlapping markers.
    
    Parameters:
        thresholds -- a list of (name, x, colour) tuples
        delta -- a float giving the minimum spacing needed between
                 adjacent threshold lines
        info -- None if no logging info should be emitted, or a dict
                with keys: "error" "balance" "qtl"
    Returns:
        adjustedThresholds -- a list akin to the input thresholds
                              but with modifications to the x value
                              to prevent overlap
    '''
    if len(thresholds) <= 1:
        return thresholds
    
    # Drop None values
    thresholds = thresholds[thresholds[:,1] != None]
    if len(thresholds) == 0:
        return []
    
    # Sort by original x-coordinate
    thresholds = thresholds[thresholds[:, 1].argsort()]
    
    # Cluster into groups where the gaps do not exceed delta
    diffs = np.diff(thresholds[:,1])
    splitAt = np.where(diffs > delta)[0] + 1
    clusters = np.split(thresholds, splitAt)
    
    # Adjust x values for each group
    adjusted = []
    
    for group in clusters:
        # Single-member clusters need no adjustment
        if len(group) == 1:
            adjusted.append(tuple(group[0]))
            continue
        
        # Multi-member clusters get spaced around the mean
        centre = np.mean([x for _, x, _ in group])
        
        for i, (name, originalX, colour) in enumerate(group):
            offset = (i - (len(group) - 1) / 2) * delta
            newX = centre + offset
            adjusted.append((name, newX, colour))
            if info:
                error, balance, qtl = info["error"], info["balance"], info["qtl"]
                error = int(error * 100)
                balance = int(balance * 100)
                print(f"QTL #{qtl} barplot with phenotype error '{error}%' and " + 
                      f"balance '{balance}%' had the {name} threshold visually " + 
                      f"adjusted to prevent overlap; actual value is {originalX} " +
                      f"but line was plotted at {newX}")
    
    return adjusted

def plot_report(locations, configuration):
    SIGNAL_COLUMNS = ["no_signal", "weak_signal", "moderate_signal", "strong_signal"]
    NUM_X_TICKS = 8
    
    if not (os.path.isfile(locations.rawTSV) and os.path.isfile(locations.rawTSV + locations.OKAY_SUFFIX)):
        raise FileNotFoundError(f"Raw results file '{locations.rawTSV}' must exist with an associated " + 
                                f"'{Locations.OKAY_SUFFIX}' flag file for a plot to be produced.")
    
    # Parse the tabulated data and derive some global parameters
    reportDF = pd.read_csv(locations.rawTSV, sep="\t")
    bootstraps = reportDF.iloc[0][SIGNAL_COLUMNS].sum()
    minPopSize = reportDF["pop_size"].min()
    maxPopSize = reportDF["pop_size"].max()
    
    # Derive threshold marker line details
    "This is the minimum spacing needed between each marker line to maintain visibility"
    DELTA = int(np.ceil((maxPopSize - minPopSize) * 0.005))
    
    # Format a composite plot
    foundMilestones = {}
    for qtlNum in reportDF["qtl"].unique():
        # Derive file names
        outputPNG = locations.outputPNG(qtlNum)
        outputPDF = locations.outputPDF(qtlNum)
        
        ## Do not skip if the file already exists, as 1) we should be able to
        ## safely allow this, and 2) we want the milestones calculations for
        ## the final step of popquis
        # if os.path.isfile(outputPNG) and os.path.isfile(outputPDF):
        #     print(f"# Stacked barplot for QTL #{qtlNum} already exists; skipping...")
        #     continue
        
        # Setup figure object
        ncol = int(len(configuration.phenotypeError) / 2) # 2==faceted rows; should result in 3 columns
        nrow = int(len(configuration.popBalance) * 2) + 1 # +1 is a spacing row; should result in 11 rows
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol,
                                 figsize=(20, 12),
                                 dpi=300,
                                 constrained_layout=False)
        fig.subplots_adjust(
            left=0.05,
            bottom=0.06,
            top=0.95,
            right=0.90,
            hspace=0.15,
            wspace=0.08
        )
        
        # Blank the middle spacing row
        for colIndex in range(ncol):
            axes[len(configuration.popBalance), colIndex].axis("off")
        
        # Add titles above each phenotypeError plot group
        groupTopRows = [0, len(configuration.popBalance) + 1]
        for i, phenotypeError in enumerate(configuration.phenotypeError):
            colIndex = i % ncol
            facetRow = i // ncol
            
            axes[groupTopRows[facetRow], colIndex].set_title(
                f"Phenotype error: {int(phenotypeError*100)}%",
                fontsize=14,
                pad=10
            )
        
        # Iterate over each configuration combination
        markers = {}
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
                foundMilestones[(qtlNum, phenotypeError, popBalance)] = milestones
                
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
                thresholds = np.array([
                    ("Weak", milestones["atleast_weak"], "white"),
                    ("Moderate", milestones["atleast_mid"], "black"),
                    ("Strong", milestones["strong_signal"], "#FFFF00")
                ], dtype=object)
                thresholds = _separate_thresholds(thresholds, DELTA, # prevent overlap
                                                  info={
                                                      "error": phenotypeError,
                                                      "balance": popBalance,
                                                      "qtl": qtlNum
                                                  }) 
                
                for labelStr, xCoord, colourStr in thresholds:
                    marker = ax.vlines(xCoord,
                                       ymin=0, ymax=bootstraps,
                                       linestyle="--", color=colourStr,
                                       label=labelStr)
                    markers.setdefault(labelStr, marker)
                
                # Set axis limits
                ax.set_ylim(0, bootstraps)
                ax.set_xlim(minPopSize, maxPopSize)
                
                # Set axis aesthetic
                ax.set_xticks(np.asarray(
                    np.linspace(minPopSize, maxPopSize, num=NUM_X_TICKS),
                dtype=np.int64))
                
                ax.set_yticks([0, bootstraps//2, bootstraps],
                              labels=["", "50%", "100%"]) # don't show 0% to prevent text overlap
                
                # Turn off axis labels if applicable
                if x != len(configuration.popBalance) - 1:
                    ax.set_xticklabels([])
                if colIndex != 0:
                    ax.set_yticklabels([])
        markers = list(markers.values())
        
        # Configure the legend
        fig.legend(
            title="Signal Strength",
            facecolor="lightgrey",
            edgecolor="black",
            handles=stack,
            loc="lower left",
            bbox_to_anchor=(0.91, 0.5105, 0.08, 0.15), # right side slightly above middle
            mode="expand"
        )
        
        fig.legend(
            title="Signal Threshold",
            facecolor="lightgrey",
            edgecolor="black",
            handles=markers,
            loc="upper left",
            bbox_to_anchor=(0.91, 0.35, 0.08, 0.15), # right side slightly below middle
            mode="expand"
        )
        
        # Write output files
        fig.savefig(outputPNG)
        fig.savefig(outputPDF)
    
    # Return the milestones for separate tabulated output
    return foundMilestones

def write_thresholds_tsv(locations, configuration, foundMilestones):
    THRESHOLD_KEYS = ["atleast_weak", "atleast_mid", "strong_signal"]
    FORMATTED_KEYS = ["weak", "moderate", "strong"] # prettier for presentation
    
    with open(locations.thresholdsTSV, "w") as fileOut:
        fileOut.write(f"pop_balance\tphenotype_error\tqtl\tthreshold\tnum_samples\n")
        
        for (popBalance, phenotypeError), popSizes in configuration:
            if popSizes is None:
                continue
            
            qtlNum = 0
            while True:
                qtlNum += 1
                
                try:
                    milestones = foundMilestones[(qtlNum, phenotypeError, popBalance)]
                except:
                    break
                
                for dictKey, displayKey in zip(THRESHOLD_KEYS, FORMATTED_KEYS):
                    value = milestones[dictKey]
                    fileOut.write("\t".join(map(str, [
                        popBalance, phenotypeError, qtlNum, displayKey, value
                    ])) + "\n")
