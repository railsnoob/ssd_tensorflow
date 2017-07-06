import pickle
import os
from ssd_config import *
import numpy as np
import subprocess
from ssd_pre_process import SSDPreProcess

if __name__ == "__main__":

    # create_data_set(90,"t90")
    # create_data_set(10,"t10")
    # stanford = SSDPreProcess("stanford")
    # pre.create_data_set(3000,"t3000")

    # TODO maybe this hsould also just look at the directory?
    # pre = SSDPreProcess("stanford","data")

    pre = SSDPreProcess("voc2012","voc-data/VOC2012",SSDConfig("./tiny_voc"))

    # TODO Do I need to name this again?
    pre.create_data_set(10)
    
    # pre.calc_default_box_sizes()
    

    
