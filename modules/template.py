# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

from scipy.optimize import minimize_scalar

class Template:
    '''
    Methods:
        split_gaussian_template -- produces a numpy array with a Guassian distribution
                                      of numbers from 0 to 1
        fit -- fits a numpy array of statistical values against a Guassian template
               to measure its correlation to the template shape and how "pointy" that
               template is.
    '''
    @staticmethod
    def split_gaussian_template(length, peakFraction=0.20, left=True):
        if not (0 < peakFraction < 1):
            raise ValueError("peakFraction must be between 0 and 1")
        
        x = np.arange(length)
        centre = length // 2
        
        fwhm = peakFraction * length # how wide should our FWHM be
        sigma = fwhm / 2.355 # 2.355 is an approximate constant related to the FWHM
        
        template = np.exp(
            -((x - centre) ** 2) /
            (2 * sigma**2)
        )
        if left:
            return template[:centre]
        else:
            return template[centre:]
    
    @staticmethod
    def _correlation(y, template):
        '''
        Calculate the Pearson product-moment correlation coefficient between an array of
        values (e.g., Euclidean distance floats) and a Guassian distribution with variable
        width. The result is a value ranging from -1 to +1 representing negative or positive
        correlation strength between the values and the Guassian distribution shape.
        '''
        try:
            if y.shape != template.shape:
                raise ValueError("Shape of y and template arrays do not match")
        except AttributeError:
            raise ValueError("y and/or template are not numpy arrays and have no .shape attribute")
        
        if np.isclose(y.std(), 0):
            return 0
        if np.isclose(template.std(), 0):
            return 0
        
        return np.corrcoef(y, template)[0, 1]
    
    @staticmethod
    def fit_gauss(y, minimum=0.01, maximum=0.5):
        '''
        Fit a numpy array of numeric values against a QTL-like peak as modelled through
        an optimised Guassian function. This function is optimised for a FWHM proportion
        ranging from minimum -> maximum.
        
        The resulting score is the Pearson correlation from -1 to +1 indicating how well
        the number distribution matches against a modelled Guassian shape. It additionally
        returns the FWHM proportion that provided this optimised Pearson correlation value.
        In this respect, a lower value means the QTL is more of a "spike" and a larger
        value indicates a more gradual slope towards the edges.
        
        Parameters:
            y -- a numpy array of numeric values to have fitted against a Guassian distribution
            minimum -- a float value giving the minimum proportion of the data distribution
                       to be part of the FWHM; default == 0.01 which is 1%
            maximum -- a float value giving the maximum proportion of the data distribution
                       to be part of the FWHM; default == 0.5 which is 50%
        Returns:
            correlation -- a float of the Pearson correlation
            width -- a float of the FWHM between minimum -> maximum that provided optimised
                     fitting
        '''
        def _objective(peakFraction, left):
            template = Template.split_gaussian_template(
                len(y),
                peakFraction,
                left=left
            )
            if left:
                return -Template._correlation(y[:len(y)//2], template)
            else:
                return -Template._correlation(y[len(y)//2:], template)
        
        if not (0 < minimum < 1):
            raise ValueError("Template.fit minimum must be between 0 and 1")
        if not (0 < maximum < 1):
            raise ValueError("Template.fit maximum must be between 0 and 1")
        if not minimum < maximum:
            raise ValueError("Template.fit maximum must be greater than minimum")
        
        leftResult = minimize_scalar(
            _objective,
            bounds=(minimum, maximum),
            args=(True), # left is True
            method="bounded"
        )
        rightResult = minimize_scalar(
            _objective,
            bounds=(minimum, maximum),
            args=(False), # left is False; look at right side instead
            method="bounded"
        )
        
        leftCorrelation = -leftResult.fun
        rightCorrelation = -rightResult.fun
        
        leftWidth = leftResult.x
        rightWidth = rightResult.x
        
        return leftCorrelation, rightCorrelation, leftWidth, rightWidth
    
    @staticmethod
    def generate_focus_template(y, transitionControl=2):
        '''
        Generates a template with a weighting/scoring schema for
        half of a QTL distribution, going from lowest at left to
        highest at right; flip the input array first if it is
        from the right-hand side of a distribution. An example
        output might be like:
        
        transitionControl(2) [-1, -0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75, 1]
        transitionControl(4) [-1, -0.83, -0.66, -0.50, -0.33, -0.16, 0, 0.50, 1]
        
        Parameters:
            y -- a numpy array of numeric values representing half of a QTL
                 distribution
            transitionControl -- an integer giving the (1 / transitionControl)
                                 distance away from the QTL centre where maximum
                                 values change from being penalised to rewarded
        '''
        transition = len(y) // transitionControl
        
        penaltyZone = np.linspace(-1, 0, num=len(y)-transition) # will have -1 taken off its length
        rewardZone = np.linspace(0, 1, num=transition+1) # put the +1 here
        
        return np.concatenate((penaltyZone[:-1], rewardZone)) # this prevents a sequential [0,0] in the template
    
    def _focusing(y, template):
        maxIndices = np.flatnonzero(y == y.max())
        focalPoints = template[maxIndices]
        return np.sum(focalPoints) / len(focalPoints)
    
    @staticmethod
    def fit_focus(y):
        '''
        Fits a numpy array of numeric values against a Template that models the
        importance of where a maximum value has occurred in terms of how it will
        direct the human focus to that point as being an important region. Maximum
        values close to the centre of the QTL should be scored positively as
        it draws focus to the true QTL region, whereas regions further away from
        the true QTL should be scored in a progressively negative manner as they
        actively draw focus away from where it should be.
        
        Parameters:
            y -- a numpy array of numeric values to have compared against the
                 focus scoring function
        Returns:
            leftFocus / rightFocus -- a float value from -1 to +1 indicating how
                                      well the left and right side of the QTL
                                      distribution draw focus to the true QTL
                                      location
        '''
        leftY = y[:len(y)//2]
        rightY = np.flip(y[len(y)//2:])
        
        leftTemplate = Template.generate_focus_template(leftY)
        rightTemplate = Template.generate_focus_template(rightY)
        
        leftFocus = Template._focusing(leftY, leftTemplate)
        rightFocus = Template._focusing(rightY, rightTemplate)
        
        return ((leftFocus * (np.max(leftY) / np.max(y))),  # scale the focus by its proportion
               (rightFocus * (np.max(rightY) / np.max(y)))) # of actually containing a global max
