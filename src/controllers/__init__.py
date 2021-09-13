REGISTRY = {}

from .basic_controller import BasicMAC, IndependentMAC
from .cqmix_controller import CQMixMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["independent_mac"] = IndependentMAC
REGISTRY["cqmix_mac"] = CQMixMAC