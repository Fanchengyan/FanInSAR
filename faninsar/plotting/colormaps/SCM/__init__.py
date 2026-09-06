# Import the new SCM colormap loader
from .colormaps import __all__, __dir__, __getattr__, names, version

# For backward compatibility, expose all colormap names at module level
# The __getattr__ function from colormaps.py will handle dynamic loading
