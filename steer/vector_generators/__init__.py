"""
vector_generators/__init__.py
==============================
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  sta and sae_feature were commented out (disabled).
          reps, sft, spilt removed entirely — added upstream after you
          cloned; confirmed absent by the audit report (DELETED section).
          Importing them would crash on startup with ModuleNotFoundError.
"""

from .caa import *
from .lm_steer import *
from .vector_prompt import *
# from .sta import *          # exists but not used in your experiment
# from .sae_feature import *  # exists but not used in your experiment
from .merge import *
from .vector_generators import *
