import sys
import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import pickle
from ssd_config import SSDConfig

from utils import print_stats2
from utils import non_max_suppression_fast

print("Using TF Version",tf.__version__)


class Inference:
    def __init__(self, dirname):
        self.dirname = dirname
        self.cfg = SSDConfig(dirname)
    
    def get_box_lefttop_and_rightbottom(self, center_x,center_y,w,h):
        """ Calculates the left_top and right_bottom coordinates as required by cv2 to plot box.
        Args: center x,y coordinates , width & height
        Returns: left_top & right_bottom coordinates as arrays
        """
        left_top = [ center_x - w/2, center_y - h/2 ]
        right_bottom = [ center_x + w/2, center_y + h/2 ]
        return left_top + right_bottom

    
    def get_coordinates(self,norm_box,feat_map,row_i,col_i,default_box_scale):
        """ Convert from normalized predictions to picture coordinates """
        import numpy as np
    
        ## TODO This could be a bug ... Mixing up Width and Height
        cell_w = self.cfg.g("image_width")/feat_map[0]
        cell_h = self.cfg.g("image_height")/feat_map[1]
    
        dbox_cx_real = (row_i + 0.5 +default_box_scale[0] )*cell_w
        dbox_cy_real = (col_i + 0.5 + default_box_scale[1])*cell_h
    
        dbox_width   = cell_w * default_box_scale[2]
        dbox_height  = cell_h * default_box_scale[3]
    
        box_center_x = norm_box[0]*dbox_width + dbox_cx_real 
        box_center_y = norm_box[1]*dbox_height + dbox_cy_real
    
        box_w        = np.exp(norm_box[2])*dbox_width
        box_h        = np.exp(norm_box[3])*dbox_height

        return self.get_box_lefttop_and_rightbottom(box_center_x,box_center_y,box_w,box_h)


    def convert_coordinates_to_boxes(self,loc,conf,probs):
        import numpy as np
        boxes = []
        confs = []
        ii = 0 # index on the y_pred_conf
        print("LOC=",np.where(loc!=0.0) )
        print("CONF=",np.where(conf != 0.0) )
        for feat_map in self.cfg.g("feature_maps"):
            for feature_map_row in range(feat_map[0]):
                for feature_map_col in range(feat_map[1]):
                    for default_box_scale in self.cfg.g("default_box_scales"):
                        if probs[0][ii] > self.cfg.g("pred_conf_threshold") and conf[0][ii] > 0.0:
                            print("Detected at",feature_map_row, feature_map_col,"at featmap (",feat_map,")")
                            box = self.get_coordinates(loc[0][ii*4:ii*4+4],feat_map,feature_map_row,feature_map_col,default_box_scale)
                            boxes.append(box)
                            confs.append([probs[0][ii],conf[0][ii]])
                        ii+=1
        return boxes, confs


    # accuracy
    # Run through feed-forward
    # Calculate the number of boxes > threshold versus number of false 
    # correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # accuracy_conf = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy_conf')
    # accuracy_loc = # calculate the number thats 50% sized coverage.

    def _dump_tensors(self):
        tensors = [n.name for n in tf.get_default_graph().as_graph_def().node ]
        print(tensors)
        
    def run_inference(self,image_name, model_name="trained-model"):
    
        with tf.Graph().as_default(), tf.Session() as predict_sess:
            saver = tf.train.import_meta_graph(self.dirname + "/" + model_name + ".meta")
            saver.restore(predict_sess,self.dirname + "/" + model_name)

            x            = tf.get_default_graph().get_tensor_by_name("x:0")
            y_pred_conf  = tf.get_default_graph().get_tensor_by_name("y_predict_conf:0")
            y_pred_loc   = tf.get_default_graph().get_tensor_by_name("y_predict_loc:0")
    
            img          = mpimg.imread(self.cfg.g("images_path")+"/" +image_name)
            
            if img.shape[0] != self.cfg.g("image_height") or img.shape[1] != self.cfg.g("image_width"):
                img      = cv.resize(img,(self.cfg.g("image_height"),self.cfg.g("image_width")))
                
            img          = (img-128)/128

            print("run_inference():imp.shape=",img.shape)
        
            batched_img         = np.reshape(img, [-1,self.cfg.g("image_height"),self.cfg.g("image_width"),self.cfg.g("n_channels")])
            fdict               = {x:batched_img}

            y_pred_conf_tmp     = tf.reshape(y_pred_conf,[-1,self.cfg.g("num_preds"),2])
            all_probabilities   = tf.nn.softmax(y_pred_conf_tmp)
            probs1, preds_conf1 = tf.nn.top_k(all_probabilities)

            print("PROBS1",probs1)
            print("PREDS_CONF1",preds_conf1)
            
            probs               = tf.reshape(probs1,[-1,self.cfg.g("num_preds")])
            preds_conf          = tf.reshape(preds_conf1,[-1,self.cfg.g("num_preds")])
            
            print("PREDS_CONF", preds_conf)
            print("Y_PRED_LOC", y_pred_loc)
            
            predicted_conf, predicted_loc, predicted_probs,y_p_conf_out,probs1_out, preds_conf1_out= predict_sess.run([preds_conf,\
                                    y_pred_loc,\
                                    probs,\
                                    y_pred_conf,\
                                    probs1,\
                                    preds_conf1],feed_dict=fdict)
                                    
        # DEBUG OUTPUT
        print("y_p_conf_out.shape",y_p_conf_out.shape)
        print_stats2(y_p_conf_out,"y_pred_conf")
        print_stats2(preds_conf1_out,"preds_conf1")
        print("probs1.shape",probs1.shape)
        print("y_pred_loc ",predicted_loc.shape)
        print("y_pred_conf ",predicted_conf.shape)    
        print("preds_conf1",preds_conf1.shape)
        print("preds_conf.shape",preds_conf.shape)
        print("probs.shape",probs.shape)
    
        return predicted_conf, predicted_loc, predicted_probs

    def debug_draw_boxes(self, img, boxes ,color, thick):
        for box in boxes:
            pts = [ min(max(0,int(v)), 640 if i==0 or i==2 else 480) for i,v in enumerate(box)]
            cv.rectangle(img,(pts[0],pts[1]),(pts[2],pts[3]),color,thick)

        
    def predict_boxes(self,model_name="trained-model"):
        """ Given a directory containing the dataset and images_path show objects detected for
        a random image.

        Args    - dirname: Name of directory containing meta data of training run.
        Returns - Nothing. Shows the pic with detections.
    
        """ 
        images_path            = self.cfg.g("images_path")
    
        train_imgs             = pickle.load(open(self.dirname+"/train.pkl","rb"))
        
        image_info             = random.choice(list(train_imgs))
        image_name             = image_info['img_name']
        p_conf, p_loc, p_probs = self.run_inference(image_name,model_name)
        non_zero_indices       = np.where(p_conf > 0)[1]

        # DEBUGGING
        print("p_conf={} p_loc={} p_probs={}".format(p_conf.shape,p_loc.shape,p_probs.shape))
        print("Non zero indices",non_zero_indices)
        for i in non_zero_indices:
            print(i,") location",p_loc[0][i*4:i*4+4],"probs", p_probs[0][i],"conf",p_conf[0][i])
    
        boxes, confs = self.convert_coordinates_to_boxes(p_loc,p_conf,p_probs)
        print("Boxes")
        for i,a in enumerate(zip(boxes,confs)):
            print i,a
    
        boxes = non_max_suppression_fast(boxes,0.3)
        
        img   = mpimg.imread(images_path+"/"+image_name)
    
        self.debug_draw_boxes(img,boxes,(0,255,0),2)
    
        plt.figure(figsize=(8,8))
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    if (len(sys.argv) < 3 ):
        print("Please supply a directory name contraining a configuration file.")
        print("Usage:")
        print("{} <directory-containing-configuration-yaml-file> <model-name-relative-to-directory>".format(sys.argv[0]))
        print("Example:")
        print("{} {} {}".format(sys.argv[0],"/Users/vivek/work/ssd-code/tiny_voc","Jul_05_161614_O3K2T/final-model"))
        sys.exit()
        
    inf = Inference(sys.argv[1])
    inf.predict_boxes(sys.argv[2]) # TODO: find the last model by default
