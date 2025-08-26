r"""A kernel module that contains a global register for unified model, dataset, and OOD algorithms access.
"""

class Register(object):
    r"""
    Global register for unified model, dataset, and OOD algorithms access.
    """

    def __init__(self):
        self.datasets = dict()
       

    def dataset_register(self, dataset_class):
        r"""
        Register for dataset access.

        Args:
            dataset_class (class): dataset class

        Returns (class):
            dataset class

        """
        self.datasets[dataset_class.__name__] = dataset_class
        return dataset_class

register = Register()  #: The GOOD register object used for accessing models, datasets and OOD algorithms.