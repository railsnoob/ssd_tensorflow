from stanford_pedestrian_data import StanfordPedestrianData
from voc_data import VOCData

# This class knows what dataset is where.
class DatasetFactory:

    def __init__(self):
        pass
    
    def get_dataset_interface(dataset_name,data_dir):
        """
        Static function which returns an interface to a dataset.
        """
        if dataset_name == "stanford":
            return StanfordPedestrianData(data_dir)
        elif dataset_name == "voc2012":
            return VOCData(data_dir)
        else:
            raise "Unknown dataset: {}".format(dataset_name)
