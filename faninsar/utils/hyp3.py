import pandas as pd
import sys
from pathlib import Path

try:
    import hyp3_sdk as sdk
except:
    print('hyp3_sdk not installed')
    print('install with: pip install hyp3_sdk')
    print('or: conda install -c conda-forge hyp3_sdk -y')



