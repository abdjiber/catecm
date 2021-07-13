class ModelNotFittedError(Exception):
    """A class exception raising an error if a model is not fitted"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class CatECMWarning(UserWarning):
    pass