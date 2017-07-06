
import sys
import numpy as np
import tensorflow as tf
import random
import pickle

from ssd_config import SSDConfig
from net_factory import NetFactory

import cv2 as cv

import matplotlib.image as mpimg
from PIL import Image

imgs = {}

class SSDTrain:

    def __init__(self,dirname):
        """
        Args 
        dirname : Name of the directory containing the yaml file containing all meta-data
        """
        self.cfg     = SSDConfig(dirname)
        self.dirname = dirname
        self._net    = NetFactory.get_net(self.cfg.g("net"))(num_default_boxes= self.cfg.g("num_default_boxes"),
                                                             num_classes= self.cfg.g("num_classes"))
        print("Using NET={} DATASET={}".format(self.cfg.g("net"),self.cfg.g("dataset")))
        

    def _get_yconf_mask(self, y_conf, y_loc, n_matched):
        """
        We only use neg_post_ratio times the number of positive gt matches.
        This is not needed for y_loc as y_loc losses are only for matched default boxes that match ground_truth box
        yconf_mask will contain a 1 for a data point that we will use and 0 for one that we will not. 
        Args
        y_conf = array of confidences
        y_loc  = array of location
        n_matched = number of matched gt boxes

        Returns
        A mask which specifies the total number of confidences to be used in calculating loss.
        """
        yconf_mask     = np.array(y_conf, copy = True)
        zero_indexes   = np.where(yconf_mask == 0)[0]
        n_negatives    = n_matched*self.cfg.g("neg_pos_ratio")
        yconf_mask     = np.minimum(yconf_mask,1)
        the_chosen_few = random.sample(zero_indexes.tolist(),n_negatives)

        print("n_negatives=",n_negatives);    print("zero_indexes=",zero_indexes.shape)

        for i in the_chosen_few:
            yconf_mask[i] = 1

        return yconf_mask

    def _get_image(self, fname):
        """
        Args fname : filename of image

        Returns the image as array with dim (w,h,channels) """
        img = imgs.get(fname)

        if  img == None:
            img = mpimg.imread(fname)
            img = (img-128)/128
            imgs[fname] = img
    
        return img
    
    def _batch_gen(self,start,end, batch_size,data,num_conf,num_loc,images_path):
        """
        Args: start - starting position in data
              end - ending position in data
              batch_size - batch size
              data - contains image_name, y_loc, y_conf, n_matched for each image
              num_conf - total number of detection confidences
              num_lock - total number of location confidences
              images_path - path to find images
        """
        import pickle
        import numpy as np

        X_train = np.zeros([batch_size,self.cfg.g("image_height"),self.cfg.g("image_width"),self.cfg.g("n_channels")])
        Y_conf           = np.zeros([batch_size,num_conf])
        Y_conf_loss_mask = np.zeros([batch_size,num_conf])
        Y_loc            = np.zeros([batch_size,num_loc])
        n_matched        = np.zeros([batch_size,1])
    
        for i,m in enumerate(data[start:end]):
            img            = self._get_image(images_path+"/"+m['img_name'])

            if img[0] != self.cfg.g("image_height") or img[1] != self.cfg.g("image_width"):
                img            = cv.resize(img,(self.cfg.g("image_height"),self.cfg.g("image_width")))
                
            X_train[i]     = img
            Y_loc[i]       = m['y_loc']
            Y_conf[i]      = m['y_conf']
            Y_conf_loss_mask[i]= self._get_yconf_mask(m['y_conf'],m['y_loc'], m['n_matched'])
            n_matched[i]   = m['n_matched']
        
        return X_train, Y_loc, Y_conf, n_matched,Y_conf_loss_mask

          
    def _len_data(self,dirname,batch_type):
        training_data = pickle.load(open(dirname+"/"+batch_type+".pkl","rb"))
        return len(training_data)

    def _evaluate_testset(self, dirname,sess):
        """ TODO ... using val loss for now. Run evaluate the test set """
        gen = batch_gen("test",dirname)
        sess.run(accuracy)

    def _evaluate_validation(self,dirname,y_val_loc,y_val_conf,y_pred_loc,y_pred_conf, sess):
        """TODO using val loss for now """
        batch_gen("val",dirname)
        for a in range(B):
            pass
        sess.run(accuracy)
        # Find the difference conf
        # find the difference between loc
        # Use the model to predict bb's 
        # NBS to filter out 
        # If overlap within THRESHOLD then accurate. 
        pass


    def _print_stats2(self,the_y,name,action="not"):
        indices = None
        if action=="not":
            indices = np.where(the_y != 0.0)[1]
        if action=="gt":
            indices = np.where(the_y > 0.00001)[1]
        print(name+" indices=",indices)
        if action=="not":
            print(name," != 0.0", the_y[0][indices]  )
        if action=="gt":
            print(name," > 0.00001", the_y[0][indices]  )    
        print(name+" indices", indices.shape )

        

    def _smooth_l1(self,x):
        return tf.where( tf.less_equal(tf.abs(x),1.0), 0.5*x*x,  tf.abs(x) - 0.5)
    
    def _ssd_graph(self,x,y_loc,y_conf,num_matched,y_conf_loss_mask):
        ## CREATE THE GRAPH
        y_predict_loc, y_predict_conf = self._net.graph(x)
        
        y_predict_loc1 = y_predict_loc
        y_predict_conf1 = y_predict_conf
    
        dbg_num_conf_mask = y_predict_conf
        dbg_num_loc = y_predict_loc
        
        y_conf_loss_mask = tf.cast(y_conf_loss_mask,tf.float32)
        
        y_predict_conf = tf.reshape(y_predict_conf,[-1,self.cfg.g("num_preds"),self.cfg.g("num_classes")])
        print("  predict_conf.shape & y_conf.shape ",y_predict_conf.shape, y_conf.shape)
        
        Lconf = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_predict_conf, labels=y_conf)
        Lconf2 = y_conf_loss_mask * Lconf
        Lconf = tf.reduce_sum(Lconf2)
        y_conf_1_column = y_conf

        print("num_matched size, NUM_LOC, NUM_CONF ====",num_matched,self.cfg.g("num_loc"),self.cfg.g("num_conf"))
    
        # Take the confidence matrix and convert it into 1s and 0s representing whether we should zero out the box locations or not. 
        matching_box_present_mask = np.array([])    
        matching_box_present_mask = tf.minimum(y_conf,1)
        matching_box_present_mask = tf.concat([y_conf,y_conf,y_conf,y_conf],1)
        print("y_repdict_conf y_predict_conf.shape[1]*4",y_conf.shape[1],y_predict_conf.shape[1]*4)
        print(matching_box_present_mask)
        # matching_box_present_mask = tf.reshape(matching_box_present_mask,[-1,y_predict_conf.shape[1]*4])
        
        matching_box_present_mask = tf.cast(matching_box_present_mask,tf.float32)

        print("matching_box_present_mask shape ====", matching_box_present_mask.shape,"y_loc.shape shape ====", matching_box_present_mask.shape)
    
        Lbox_coords = self._smooth_l1(y_loc - y_predict_loc)
        Lbox_coords = tf.multiply(matching_box_present_mask, Lbox_coords) # Y_conf will already be zero
        Lbox_coords_before_sum = Lbox_coords # should have same coordinates as y_conf and n*4 non zero values
        Lbox_coords = tf.reduce_sum(Lbox_coords)
        # total_loss = (1/num_matched[0])*Lbox_coords + Lconf
        total_loss = Lbox_coords + Lconf
        optimizer = tf.train.AdamOptimizer() # TODO allow changing initial learning_rate value
        training_operation = optimizer.minimize(total_loss)
        saver =  tf.train.Saver()
        tf.add_to_collection("x", x)
        tf.add_to_collection("y_predict_loc", y_predict_loc)
        tf.add_to_collection("y_predict_conf", y_predict_conf)
        tf.add_to_collection("y_predict_loc1", y_predict_loc1)
        tf.add_to_collection("y_predict_conf1", y_predict_conf1)
        
        debug_stats = { "Conf-Loss-Before-Reduce-Sum" : Lconf2,
                        "dbg_num_conf_mask": dbg_num_conf_mask,
                        "dbg_num_loc": dbg_num_loc,
                        "Lbox_coords_before_sum":Lbox_coords_before_sum,
                        "box_mask": matching_box_present_mask,
                        "Lbox_coords": Lbox_coords,
                        "Lconf": Lconf,
                        "y_predict_conf1":y_predict_conf1,
                        "y_predict_loc1":y_predict_loc1,
                        "y_predict_conf":y_predict_conf,


        }

        return saver, debug_stats, total_loss,training_operation

    def _print_debug_stats(self,debug_out):
        print("OUTPUT VARS ==============================")
        self._print_stats2(debug_out["Conf-Loss-Before-Reduce-Sum"],"Conf Loss Before Reduce Sum")
        self._print_stats2(debug_out["y_predict_conf"],"y_predict_conf","gt")
        self._print_stats2(debug_out["dbg_num_loc"],"dbg_num_loc")
        self._print_stats2(debug_out["box_mask"],"box_mask")
        self._print_stats2(debug_out["Lbox_coords_before_sum"],"Lbox_coords_before_sum")

    def _calc_validation_losses(self, sess, epoch_i, train_loss,batch_size,valid_data,x,y_conf,y_loc,num_matched, y_conf_loss_mask, total_loss):

        num_valid_samples       = self._len_data(self.dirname, "val")
        batch_size              = min(num_valid_samples,batch_size)
        epoch_validation_losses = []
        
        for valid_offset in range(0, num_valid_samples,batch_size):
            start = valid_offset
            end   = valid_offset + batch_size
            X_valid_batch, y_valid_batch_loc, y_valid_batch_conf, n_valid_matched_batch, y_valid_conf_mask = self._batch_gen(start,end,batch_size, valid_data,self.cfg.g("num_conf"),self.cfg.g("num_loc"), self.cfg.g("images_path"))
                
            validation_loss = sess.run([total_loss], feed_dict={x          : X_valid_batch,
                                                                y_conf     : y_valid_batch_conf,
                                                                y_loc      : y_valid_batch_loc,
                                                                num_matched: n_valid_matched_batch,
                                                                y_conf_loss_mask:y_valid_conf_mask})
            epoch_validation_losses.append(validation_loss)

        print("============")
        print("EPOCH {} ValLoss={} TrainLoss={}".format(epoch_i, validation_loss, train_loss) )
        print(self.dirname+"/"+"model-vloss-"+str(validation_loss)+"tloss"+str(train_loss)+"EPOCH-",str(epoch_i))
        print("============")

        epoch_validation_loss = np.mean(epoch_validation_losses)
        return epoch_validation_loss

    
    
    def debug_output_vars(self,debug_out,train_loss,y_batch_loc,y_batch_conf,y_conf_mask):
        print("Training LOSS = {} Lconf={} Lbox_coords={}".format(train_loss,
                                                                  debug_out["Lconf"],
                                                                  debug_out["Lbox_coords"]))
        print("ypred1 shape={} yloc1 shape={}".format(debug_out["y_predict_conf1"].shape,
                                                      debug_out["y_predict_loc1"].shape))
        self._print_stats2(y_batch_loc,"y_batch_loc")
        self._print_stats2(y_batch_conf,"y_batch_conf")
        self._print_stats2(y_conf_mask,"y_conf_mask")
        self._print_debug_stats(debug_out)

    def train_the_net(self):
        # Setup the loss for the coordinates and the bl
        cfg = self.cfg
        cfg.save_at_beginning_of_run()

        dirname = self.cfg.g("dirname")
        ## INITIALIZATION
        x                = tf.placeholder(tf.float32,(None, cfg.g("image_height"),
                                                      cfg.g("image_width"), cfg.g("n_channels")))
        y_loc            = tf.placeholder(tf.float32,(None,cfg.g("num_loc")))
        y_conf           = tf.placeholder(tf.int32,(None,cfg.g("num_conf")))
        num_matched      = tf.placeholder(tf.int32,(None,1))
        y_conf_loss_mask = tf.placeholder(tf.int32,(None,cfg.g("num_conf")))

        saver, debug_stats, total_loss, training_operation = self._ssd_graph(x,y_loc,y_conf,num_matched,y_conf_loss_mask)
        
        printed                        = tf.Print(total_loss,[total_loss])
        
        # TODO This should go intot the dataset class
        data                = pickle.load(open(dirname+"/train.pkl","rb"))
        valid_data          = pickle.load(open(dirname+"/val.pkl","rb"))

        images_path         = cfg.g("images_path");
        batch_size          = cfg.g("batch_size")
        
        all_training_losses = []
        cumulative_losses   = []
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = self._len_data(dirname,"train")
            print("\n\n\n\n Number of images",num_examples)

            for epoch_i in range(self.cfg.g("num_epochs")):

                epoch_train_losses      = []
                epoch_validation_losses = []
                batch_size              = min(num_examples,batch_size)
                
                for offset in range(0,num_examples, batch_size):
                    
                    start = offset
                    end   = offset + batch_size
                    X_batch, y_batch_loc, y_batch_conf, n_matched_batch, y_conf_mask = self._batch_gen( start,end,batch_size,data,cfg.g("num_conf"),cfg.g("num_loc"),images_path)
                    
                    print("EPOCH ==== ",epoch_i,"offset=",offset)
                    print(X_batch.shape, y_batch_loc.shape, y_batch_conf.shape, n_matched_batch, y_conf_mask.shape)
                
                    _, lprinted, train_loss, debug_out  = sess.run([ training_operation, printed, total_loss, debug_stats],
                                                                    feed_dict={x:X_batch,
                                                                    y_conf:y_batch_conf,
                                                                    y_loc:y_batch_loc,
                                                                    num_matched:n_matched_batch,
                                                                    y_conf_loss_mask:y_conf_mask})
                    
                    epoch_train_losses.append(train_loss)
                    self.debug_output_vars(debug_out,train_loss,y_batch_loc,y_batch_conf,y_conf_mask) 
                  
                    
                epoch_train_loss = np.mean(epoch_train_losses)
                all_training_losses.append([epoch_i,epoch_train_loss])
                
                pickle.dump(all_training_losses, open(cfg.g("run_dir")+"/train_losses_till_epoch-"+str(epoch_i),"wb"))

                if (epoch_i % 5 == 0):
                    # After every 5 epochs we can calculate validation loss & accuracy (TODO).
                    epoch_validation_loss = self._calc_validation_losses(sess, epoch_i,epoch_train_loss,batch_size,valid_data,x,y_conf,y_loc,num_matched, y_conf_loss_mask, total_loss)
                    epoch_train_loss      = np.mean(epoch_train_losses)
                    cumulative_losses.append([epoch_i,epoch_train_loss,epoch_validation_loss])
                    pickle.dump(cumulative_losses, open(cfg.g("run_dir")+"/cumulative_losses_till_epoch-"+str(epoch_i),"wb"))
                    saver.save(sess,"{}/model-vloss-{}-tloss-{}-EPOCH-{}".format(cfg.g("run_dir"),
                                                                                 str(epoch_validation_loss),
                                                                                 str(epoch_train_loss),
                                                                                 str(epoch_i)))

            saver.save(sess,cfg.g("run_dir")+"/final-model")

            

    def debug_train_setup(self):
        """ Use this SOLELY to figure out the size of the feature maps. """

        x = tf.placeholder(tf.float32,shape=(None,\
                                            self.cfg.g("image_height"),\
                                            self.cfg.g("image_width"),\
                                            self.cfg.g("n_channels")),\
                                            name='x')
                                            
        y = tf.placeholder(tf.int32,shape=(None,self.cfg.g("num_preds")),name='y')
        one_hot_y = tf.one_hot(y,10)
    
        loc, conf = self._net.graph(x)
    
        # This is just a placeholder cost
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=conf,labels=y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    
    def train_dataset(self):
        ssd_train = SSDtrain(self.cfg)
        ssd_train.debug_train_setup()
    

if __name__ == "__main__":
    # train_dataset("/Users/vivek/work/ssd-code/tiny",1 )
    # vgg = SSDTrain("/Users/vivek/work/ssd-code/tiny")
    # vgg.debug_train_setup()

    voc_alex_net = SSDTrain("/Users/vivek/work/ssd-code/tiny_voc")
    voc_alex_net.train_the_net()

    
