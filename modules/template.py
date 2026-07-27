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
    def _correlate(y, template):
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
        
        y_std = y.std()
        if y_std == 0:
            return 0
        
        return np.corrcoef(
            (y - y.mean()) / y_std,
            (template - template.mean()) / template.std()
        )[0, 1]
    
    @staticmethod
    def fit(y, minimum=0.01, maximum=0.5):
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
        def _objective(peakFraction, isLeft):
            template = Template.split_gaussian_template(
                len(y),
                peakFraction,
                left=isLeft
            )
            if isLeft:
                return -Template._correlate(y[:len(y)//2], template)
            else:
                return -Template._correlate(y[len(y)//2:], template)
        
        if not (0 < minimum < 1):
            raise ValueError("Template.fit minimum must be between 0 and 1")
        if not (0 < maximum < 1):
            raise ValueError("Template.fit maximum must be between 0 and 1")
        if not minimum < maximum:
            raise ValueError("Template.fit maximum must be greater than minimum")
        
        leftResult = minimize_scalar(
            _objective,
            bounds=(minimum, maximum),
            args=(True), # isLeft is True
            method="bounded"
        )
        rightResult = minimize_scalar(
            _objective,
            bounds=(minimum, maximum),
            args=(False), # isLeft is False; this isRight
            method="bounded"
        )
        
        leftCorrelation = -leftResult.fun
        rightCorrelation = -rightResult.fun
        
        leftWidth = leftResult.x
        rightWidth = rightResult.x
        
        return leftCorrelation, rightCorrelation, leftWidth, rightWidth
