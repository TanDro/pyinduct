# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib as mpl

# make everybody use the same qt version, try Qt5 first
for qt in ["PyQt5", "PyQt4"]:
    try:
        __import__(qt)
        os.environ["PYQTGRAPH_QT_LIB"] = qt
        mpl.use(qt[2:]+"Agg")
    except ImportError:
        continue

# since this is a serious toolbox
np.seterr(all="raise")

# noinspection PyUnresolvedReferences
from .core import *
# noinspection PyUnresolvedReferences
from .control import *
# noinspection PyUnresolvedReferences
from .eigenfunctions import *
# noinspection PyUnresolvedReferences
from .trajectory import *
# noinspection PyUnresolvedReferences
from .registry import *
# noinspection PyUnresolvedReferences
from .placeholder import *
# noinspection PyUnresolvedReferences
from .simulation import *
# noinspection PyUnresolvedReferences
from .shapefunctions import *
# noinspection PyUnresolvedReferences
from .visualization import *

# noinspection PyUnresolvedReferences

__author__ = "Stefan Ecklebe, Marcus Riesmeier"
__email__ = "stefan.ecklebe@tu-dresden.de, marcus.riesmeier@umit.at"
__version__ = '0.4.0'
