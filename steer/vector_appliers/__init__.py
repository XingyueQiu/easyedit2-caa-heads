"""
vector_appliers/__init__.py
===========================
MODIFIED FROM EASYEDIT2 ORIGINAL

Change 1  Removed imports for modules that don't exist in your project:
            reps, sft, spilt — added upstream after you cloned;
            confirmed absent by the audit report (DELETED section).
          sae_feature and sta are kept — they exist in your project
          even though unused by your experiment.
"""

from .caa import *
from .vector_prompt import *
from .lm_steer import *
from .merge import *
from .prompt import *
from .sta import *
from .sae_feature import *
from .vector_applier import *
