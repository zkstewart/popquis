import numpy as np

class Template:
    '''
    Args:
        length -- an integer giving the length/size of the template for matching against
                  an equivalently sized numpy array
    Attributes:
        triangle -- a numpy array of a triangle shape template (^)
        plateau -- a numpy array of a plateau-triangle-plateau shape template (_^_)
    Methods:
        _generate_templates -- private method called on object initialisation to set each template
        fit -- 
    '''
    @staticmethod
    def generate_triangle_template(length):
        '''
        Provides the expected shape of a normalised array whereby edges are zero
        and the centre is 1.
        
        Shape is akin to: ^
        
        Parameters:
            length -- an integer giving the length of an array
        Returns:
            template -- a numpy array giving the normalised shape of an array
                        that matches a triangle
        '''
        x = np.linspace(-1, 1, length)
        return 1 - np.abs(x)
    
    @staticmethod
    def generate_plateau_template(length, plateauFraction=0.25):
        '''
        Provides the expected shape of a normalised array whereby edges are zero
        and remain as zero up to plateauFraction of the length of an array before
        beginning a straight line climb to 1 at the centre.
        
        Shape is akin to: _^_
        
        Parameters:
            length -- an integer giving the length of an array
            plateauFraction -- a float giving the amount of length that should be
                               a flat (minimum) plateau at the left and right edges
        Returns:
            template -- a numpy array giving the normalised shape of an array
                        that matches a triangle with plateaus at each edge
        '''
        if plateauFraction >= 0.5:
            raise ValueError("plateauTemplate cannot have a plateau extend up to or beyond half the length of an array")
        
        # Position of inflection points of the plateau-peak-plateau shape
        left = int(length * plateauFraction)
        right = length - left
        centre = length // 2
        
        # Form the template
        template = np.zeros(length) # start->left and right->end are implicitly set as a flat zero here
        template[left:centre] = np.linspace(0, 1, num=centre-left, endpoint=False)
        template[centre:right] = np.linspace(1, 0, num=right-centre, endpoint=False)
        return template
    
    @staticmethod
    def fit(y, template):
        '''
        Calculate the Pearson product-moment correlation coefficient between an array of values
        and a corresponding array of Template'd values.
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
