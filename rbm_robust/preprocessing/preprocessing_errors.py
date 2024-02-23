class WrongInputFormat(Exception):
    def __init__(self, message="Input must be a 1D array"):
        self.message = message
        super().__init__(self.message)

class WaveletCoefficientsNotProvidedError(Exception):
    """Exception raised when the wavelet coefficients are not provided."""
    def __init__(self, message="Wavelet coefficients must be provided"):
        self.message = message
        super().__init__(self.message)

class WaveletScalesNotWellFormed(Exception):
    """Exception raised when the wavelet coefficients are not provided."""
    def __init__(self, message="Wavelet scales must have the length of 2"):
        self.message = message
        super().__init__(self.message)

