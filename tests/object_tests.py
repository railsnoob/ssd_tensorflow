# Tests classes as we build em
from ssd_pre_process import SSDPreProcess

def test_stanford_ped_dataset():
    # TODO Make sure we can get dataset directly

    # Make sre we can get it via pre_process.py
    SSDPreProcess("stanford")
    SSDPreProcess.create_data_set(100, "t100")

    # Validate that each of the things has 100 data
    
    pass
    
