import yaml
import time, random, string
import os

class SSDConfig:
    def __init__(self,dirname):
        self._c = yaml.load(open(dirname+"/ssd_config.yaml","r"))
        self._c["dirname"] = dirname
        # TODO assert that it has all the vars

        self._c["num_default_boxes"] = len(self._c["default_box_scales"])

        total = 0

        for i in self._c["feature_maps"]:
            total += i[0]*i[1]*self._c["num_default_boxes"]
            
        self._c["num_conf"]  = total
        self._c["num_preds"] = total
        self._c["num_loc"]   = total*4
        pass

    def g(self,var):
        return self._c[var]

    def _run_name(self):
        return time.strftime("%b_%d_%H%M%S_") + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    
    def save_at_beginning_of_run(self):
        # create a run_name and
        self._c["run_name"] = self._run_name()
        self._c["run_dir"] = self._c["dirname"]+"/"+self._c["run_name"]

        try:
            os.makedirs(self._c["run_dir"])
        except OSError:
            pass
        
        yaml.dump(self._c, open(self._c["run_dir"]+"/"+self._c["run_name"]+".yaml","w"))
        
