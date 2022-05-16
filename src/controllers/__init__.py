REGISTRY = {}

from .basic_controller import BasicMAC, IndependentMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["independent_mac"] = IndependentMAC