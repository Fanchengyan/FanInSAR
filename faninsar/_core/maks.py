
class Mask:
    '''A class to represent a mask.'''
    def __init__(self, mask):
        self._mask = mask
        
    @property
    def values(self):
        return self._mask
    
    @classmethod
    def from_shapefile(self, shapefile):
        pass