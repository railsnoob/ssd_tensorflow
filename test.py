import pickle
import unittest
from inference import Inference
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def debug_draw_boxes(img, boxes ,color, thick):
    for box in boxes:
        pts = [int(v) for v in box]
        cv.rectangle(img,(pts[0],pts[1]),(pts[2],pts[3]),color,thick)

class TestPreProcess(unittest.TestCase):

    def test_pre_processed_values(self):
        # Initialize y_pred_loc, conf and y_probs
        y_pred_loc = np.zeros([1,3200*4],dtype=np.float32)
        y_pred_conf = np.zeros([1,3200],dtype=np.float32)
        y_probs = list(y_pred_conf)
        
        # Read the values from the file
        p = pickle.load(open("t1/train.pkl","rb"))
        image_name = p[0]["img_name"]
        y_pred_loc = p[0]["y_loc"]
        y_pred_conf = p[0]["y_conf"]

        img = mpimg.imread("/home/ubuntu/tensorflow_ssd/data/images/"+image_name)

        inf = Inference("/home/ubuntu/tensorflow_ssd/data/")
        boxes, confs = inf.convert_coordinates_to_boxes(y_pred_loc, y_pred_conf, y_pred_conf)
        for box in boxes:
            debug_draw_boxes(img, box, (255,0,0) ,1)
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    unittest.main()
