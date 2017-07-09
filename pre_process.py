import sys
import pickle
import os
import numpy as np
import subprocess
from ssd_config import SSDConfig
from ssd_pre_process import SSDPreProcess

if __name__ == "__main__":

    # create_data_set(90,"t90")
    # create_data_set(10,"t10")
    # stanford = SSDPreProcess("stanford")
    # pre.create_data_set(3000,"t3000")

    # TODO maybe this hsould also just look at the directory?
    # pre = SSDPreProcess("stanford","data")

    if (len(sys.argv) < 5):
        print("Usage:")
        print("{} <directory-containing-configuration-yaml-file> <dataset-name> <dataset-directory> <number-of-images>".format(sys.argv[0]))
        print("Example:")
        print("{} {} {} {} {}".format(sys.argv[0],"./tiny_voc","voc2012","voc-data/VOC2012","10" ))
        sys.exit()

    
    # pre = SSDPreProcess("voc2012","voc-data/VOC2012",SSDConfig("./tiny_voc"))
    pre = SSDPreProcess(sys.argv[2] ,sys.argv[3],SSDConfig(sys.argv[1]))

    # TODO Do I need to name this again?
    pre.create_data_set(int(sys.argv[4]))
    
    # pre.calc_default_box_sizes()
    

    
