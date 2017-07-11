import random
import numpy as np
from utils import iou
import pickle
from dataset_factory import DatasetFactory


class SSDPreProcess:
    def __init__(self,dataset_name,data_dir,cfg):
        self.dataset = DatasetFactory.get_dataset_interface(dataset_name,data_dir)
        self.cfg = cfg
        pass

    def _normalize_center(self,gt,def_box_coord,def_box_wh):
        """ Normalize the center x,y coordinates separately """
        return (gt - def_box_coord)/def_box_wh


    def _normalize_wh(self,gt,def_box_l):
        """ Normalize width and height separately
        Args:
        gt = ground truth width or height
        def_box_l = default box width or height

        Returns:
        Normalized width or height which is the log of the ratio between ground truth measure and the default box measure.
        """
        return np.log(gt/def_box_l)

    def _gt_diff_defbox_normalized(self,cx_real,cy_real,dbox_w,dbox_h,box):
        """
        Get the normalized 
        """
        # center of gt
        gt_cx = box[0] + box[2]/2
        gt_cy = box[1] + box[3]/2

        # h,w of gt
        gt_w = box[2]
        gt_h = box[3]

        gt_normalized = [self._normalize_center(gt_cx,cx_real,dbox_w),\
        self._normalize_center(gt_cy,cy_real,dbox_h),\
        self._normalize_wh(gt_w,dbox_w),\
        self._normalize_wh(gt_h,dbox_h)]
        return gt_normalized

    def _get_label(self,lbl):
        """
        Return the label. Here I'm not distinguishing between person and people for now.
        """
        return self.dataset.get_label_num(lbl)

    
    def _get_default_box_real_coords(self,cx_real,cy_real,dbox_w,dbox_h):
        # Calculate real coordinates of left-top point and right-bottom using above    
        def_l_x = cx_real - dbox_w*1.0/2
        def_l_y = cy_real - dbox_h*1.0/2
        def_r_x = cx_real + dbox_w*1.0/2
        def_r_y = cy_real + dbox_h*1.0/2
        return [def_l_x,def_l_y,def_r_x,def_r_y]

    def calc_default_box_sizes(self):
        # k default boxes 4
        # m feature maps 4 
        # Figure out the sizes for default boxes based on paper
        Smin = 0.2
        Smax = 0.9
        aspect_ratios = [1.0,2.0,3.0,0.5, 0.333]
        # Sk_extra = np.sqrt()
        for k_i in range(1,5):
            Sk = Smin + (Smax - Smin)*(k_i - 1)/(4 - 1)
            for asp in aspect_ratios:
                print("K_i=",k_i," Sk=",Sk," Width=",Sk*np.sqrt(asp), "Height=",Sk/np.sqrt(asp))
        
            
    
    def pre_process_and_write_images(self,img_data,dirname,name,keys=None):
        """
        Create an output pickle file of hashes (one per image) containing: names of file, y_conf, y_loc, n_matched
        """
        num_preds = self.cfg.g("num_preds")
        
        matched_boxes = []

        if keys == None:
            keys = list(img_data.keys())

        # TODO make this into one hash for simpler tx/rx
        debug_matched_default_boxes = {}
        debug_gt_boxes = {}
        debug_default_boxes= {}
        debug_cells = {}
        debug_images = keys
    
        for img_info in keys:
            debug_matched_default_boxes[img_info] = []
            debug_gt_boxes[img_info] = []
            debug_default_boxes[img_info]= []
            debug_cells[img_info] = []
    
            y_conf = [0] * num_preds   # We only have 1 class
            y_loc = [0] * num_preds * 4 #
            has_matched_boxes = False
            n_matched = 0

            for ibox, box in enumerate(img_data[img_info]['bboxes']):
                print("BOX = ",box)
                y_index = 0
                bbox = [box[0],box[1],box[2]+box[0],box[3]+box[1]]
                debug_gt_boxes[img_info].append(bbox)
                for feat_map in self.cfg.g("feature_maps"):
                    for feature_map_row in range(feat_map[0]):
                        for feature_map_col in range(feat_map[1]):
                            for default_box_scale in self.cfg.g("default_box_scales"):
                                # width and height of a cell
                                cell_w = self.cfg.g("image_width")/feat_map[0]
                                cell_h = self.cfg.g("image_height")/feat_map[1]
                            
                                # center of cell in real coordinates (also center of dbox)
                                # cx_real = (feature_map_row + 0.5)*cell_w
                                # cy_real = (feature_map_col + 0.5)*cell_h
    
                                dbox_cx_real = (feature_map_row + 0.5 +default_box_scale[0] )*cell_w
                                dbox_cy_real = (feature_map_col + 0.5 + default_box_scale[1])*cell_h
    
                                debug_cells[img_info].append([feature_map_row*cell_w, feature_map_col*cell_h,(feature_map_row+1)*cell_w, (feature_map_col+1)*cell_h])
                                
                                # Convert offsets into real
                                dbox_real = self._get_default_box_real_coords(dbox_cx_real,
                                                                        dbox_cy_real,
                                                                        cell_w*default_box_scale[2],
                                                                        cell_h*default_box_scale[3])
                                debug_default_boxes[img_info].append(dbox_real)
                                # Note: Format is different. dbox_real is left,top coords and right,bottom coords
                                # box is left,top,width and height. 
                                iou_t = iou(dbox_real[0],
                                            dbox_real[1],
                                            dbox_real[2],
                                            dbox_real[3],
                                            bbox[0],bbox[1],bbox[2],bbox[3])
                                if(iou_t > 0.3):
                                    n_matched += 1
                                    has_matched_boxes = True
                                    debug_matched_default_boxes[img_info].append(dbox_real)
                                    y_conf[y_index] = self._get_label(img_data[img_info]['labels'][ibox])
                                    # Note in this call we pass in box instead of bbox
                                    y_loc[y_index*4 : (y_index*4+4) ] = \
                                        self._gt_diff_defbox_normalized(dbox_cx_real,
                                                                dbox_cy_real, cell_w*default_box_scale[2],
                                                                cell_h*default_box_scale[3], box)
    
                                    print(n_matched,") feat_map=",feat_map,
                                      " row=",feature_map_row," col=",feature_map_col," scale=",default_box_scale,
                                    "y_loc_entry=",y_loc[y_index*4 : y_index*4+4 ],"y_loc_x-=",y_index*4,"y-loc-y",(y_index*4+4) )
                                
                                y_index += 1
                            
            if(has_matched_boxes):
                npy_loc = np.array(y_loc,dtype=np.float32)
                print("{ y_loc non_zero:",np.where(npy_loc !=0.0)[0])
                print("{ y_loc non_zero values:",npy_loc[np.where(npy_loc != 0.0)[0]])
                npy_conf = np.array(y_conf,dtype=np.float32)
                print("{ y_conf non_zero:",np.where(npy_conf !=0.0)[0])
                print("{ y_conf non_zero values:",npy_conf[np.where(npy_conf != 0.0)[0]])
                print("n_matched:",n_matched)
                print("}")
                matched_boxes.append({
                    "img_name":img_info,
                    "y_loc":y_loc,
                    "y_conf":y_conf,
                    "n_matched":n_matched
                })

        # subprocess.call("mkdir -p {}".format(dirname),shell=True)
        pickle.dump(matched_boxes,open(dirname+"/"+name+".pkl","wb"))
    
        # everything starting with debug_ is for debugging!
        return matched_boxes, debug_gt_boxes, debug_matched_default_boxes, debug_default_boxes, debug_cells, debug_images

    

    def create_data_set(self,n):
        """ Create a data set with n test images, n/5 validation images and n/5 from test set.
        Args:
        n - Number of test images
        directory - Name of directory to create and store the splits

        Returns:
        A dictionary of the form:
        { "test": [list of test image_names],
        "train": [list of training images],
        "val": [list of validation images] }
        This function also writes this data into the main directory specified when creating this object.
        """
        test = {}
        train = []
        val = []
    
        # Make this a separate factory.
        
        glist       = self.dataset.get_train_test_splits()
        
        # train + val sets from the same training set without replacement
        total_train = n + max(n//5,1)
        # test from a separate set
        total_test  = n
    
        # TODO We should check if we are trying to sample more that whats available
        all_train   = dict((a,glist["train"][a]) for a in random.sample(list(glist["train"].keys()),total_train))
    
        train_items = list(all_train.items())
    
        train       = dict(train_items[ 0 : n ])
        val         = dict(train_items[ n : (n + max(n//5,1))])
        test        = dict((a,glist["test"][a]) for a in random.sample(list(glist["test"].keys()),total_test))

        print(train_items)
        
        self.pre_process_and_write_images(train,self.cfg.g("dirname"),"train")
        self.pre_process_and_write_images(val,self.cfg.g("dirname"),"val")
        self.pre_process_and_write_images(test,self.cfg.g("dirname"),"test")
    
        print("test len",len(test))
        print("train len",len(train))
        print("val len",len(val))
