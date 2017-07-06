import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2 as cv
from pre_process import *

def iou_tests():
    print("iou(1,2,3,4,2,1,4,3)=",iou(1,2,3,4,2,1,4,3))
    print("iou(2,1,4,3,1,2,3,4)=",iou(2,1,4,3,1,2,3,4))
    print("iou(1,2,3,4,4,2,6,4)=",iou(1,2,3,4,4,2,6,4))
    print("iou(2,1,4,3,4,2,6,4)=",iou(2,1,4,3,4,2,6,4))
    print("iou(2,1,4,3,2,1,4,3)=",iou(2,1,4,3,2,1,4,3))
    print("iou(16,144,32, 180,91, 195, 107, 223)=",iou(16,144,32, 180,91, 195, 107, 223))
    print("iou(404.0 140.0 476.0 260.0 gt= 428.90 136.16 485.20 277.99=",iou(404.0,140.0,476.0,260.0,428.90, 136.16, 485.20, 277.99));
    print("iou(404.0 140.0 476.0 260.0 404.0 140.0 476.0 260.0)",iou(404.0,140.0,476.0,260.0,404.0,140.0,476.0,260.0))
    
def test_get_default_box_real_coords():
    pass


def debug_draw_boxes(img, boxes ,color, thick):
    for box in boxes:
        pts = [int(v) for v in box]
        cv.rectangle(img,(pts[0],pts[1]),(pts[2],pts[3]),color,thick)
        

def test_pre_process_image():
    matched, debug_gt, debug_matched_default_box, dbg_def_boxes ,dbg_cells, dbg_imgs = pre_process_images("tiny/data.pkl",["img00090526.jpg"])
    img = mpimg.imread("tiny/img00090526.jpg")
    debug_draw_boxes(img, dbg_cells["img00090526.jpg"], (255,255,255),1)
    debug_draw_boxes(img, debug_gt["img00090526.jpg"], (255,0,0),1)
    debug_draw_boxes(img, debug_matched_default_box["img00090526.jpg"], (1,255,1),2)
    debug_draw_boxes(img, dbg_def_boxes["img00090526.jpg"], (1,0,255),1)
    imgplot = plt.imshow(img)
    plt.show()

def test_process_image_set(dirname):
    matched, debug_gt, debug_matched_default_box, dbg_def_boxes ,dbg_cells,dbg_imgs = pre_process_images(dirname+"/data.pkl")

    for img_name in dbg_imgs:
        print("loading ... ",img_name)
        img = mpimg.imread(img_name)
        # debug_draw_boxes(img, dbg_cells[img_name], (255,255,255),1)
        debug_draw_boxes(img, debug_gt[img_name], (255,0,0),1)
        debug_draw_boxes(img, debug_matched_default_box[img_name], (1,255,1),1)
        # debug_draw_boxes(img, dbg_def_boxes[img_name], (1,0,255),1)
        imgplot = plt.imshow(img)
        plt.show()
    
    
def test_with_images():
    matched = pre_process_images("tiny/data.pkl")
    print("Length matched =",len(matched))
    X_train = np.zeros([1,480,640,3])
    Y_conf = np.zeros([1,3])
    Y_loc = np.zeros([1,4])
    imgs = []
    for i,m in enumerate(matched):
        # img = m['img']
        # img = np.reshape(img,(1,480,640,3))
        # Draw the ground truth in red
        # print("====>",img.shape,m['gt_normalized_coords'],m['box_coords'])
        # X_train = np.vstack((X_train,img))
        print("shape=",X_train.shape)
        cnf = np.asarray(m['conf'],dtype=np.float64)
        cnf= np.reshape(cnf,(1,3))
        print("cnf shape===>",cnf.shape,Y_conf.shape)
        Y_conf = np.vstack((Y_conf,cnf))
        loc =  np.asarray(m['gt_normalized_coords'],dtype=np.float64)
        loc= np.reshape(loc,(1,4))
        print("loc shape===>",loc.shape, Y_loc.shape)
        Y_loc = np.vstack((Y_loc,loc))
        print(i,") X_train.shape = ",X_train.shape,"Y_conf",Y_conf.shape,"Y_loc",Y_loc.shape)

        # pts = [int(v) for v in m['box_coords']]
        # cv.rectangle(img,(pts[0],pts[1]),(pts[2]+pts[0],pts[1]+pts[3]),(255,0,0),1)
        # dbox = [ int(v) for v in m['def_real']]
        # # Draw the greens for matched boxes
        # if (m['conf'][0] == 1 ):
        #     # people
        #     cv.rectangle(img,(dbox[0],dbox[1]),(dbox[2],dbox[3]),(0,255,0),1)
        # elif(m['conf'][1] == 1):
        #     # person
        #     cv.rectangle(img,(dbox[0],dbox[1]),(dbox[2],dbox[3]),(0,255,255),1)
        # else:
        #     cv.rectangle(img,(dbox[0],dbox[1]),(dbox[2],dbox[3]),(200,200,200),1)
            
    print("LAST ==> X_train.shape = ",X_train.shape,"Y_conf",Y_conf.shape,"Y_loc",Y_loc.shape)
    # imgplot = plt.imshow(img)
    #plt.show()
    
    # TODO From the brute force take 1/3

from train import batch_generator
import pickle

def test_batch_gen():
    #### Batch Generator
    batch_size = 4
    dirname = "/Users/vivek/work/ssd-work/tiny"
    NUM_CONF = 3200
    NUM_LOC = 12800
    tiny_gen = batch_generator(dirname,NUM_CONF, NUM_LOC, batch_size)
    for a in range(4):
        print("=================",a,"================")
        x_train, y_loc, y_conf, n_matched, y_conf_loss_mask = next(tiny_gen)
        # number of 1s in y_conf_loss_mask = (NEG_POS_RATIO+1)*n_matched
        print("x_train.shape",x_train.shape)
        print("y_conf.shape",y_conf.shape)
        print("y_conf.1s",(y_conf[y_conf==1]).shape)
        print("y_loc.shape",y_loc.shape)
        print("y_conf_loss_mask.shape",y_conf_loss_mask.shape)
        print("y_conf_loss_mask 1s",y_conf_loss_mask[y_conf_loss_mask==1].shape)
        print("n_matched=",n_matched)
        print("Num conf mask",y_conf_loss_mask[y_conf_loss_mask == 1].shape)



def test_reconstruction_from_net_output(y_pred_loc, y_pred_conf,y_probs):
    
    pass

from inference import *

def test_reconstruction():
    y_pred_loc = np.zeros([1,3200*4],dtype=np.float32)
    y_pred_conf = np.zeros([1,3200],dtype=np.float32)
    y_probs = np.zeros([1,3200],dtype=np.float32)
    # Add first guy
    # y_index = 142
    y_pred_loc[0][142*4] = 0.0043918983288272605
    y_pred_loc[0][142*4+1]   = 0.06679562614484225
    y_pred_loc[0][142*4+2] =   0.25567246425718981
    y_pred_loc[0][142*4+3] = -0.30421435285833553
    
    y_pred_conf[0][142] = 1.0
    y_probs[0][142] = 0.9

    boxes = convert_coordinates_to_boxes(y_pred_loc,y_pred_conf,y_probs)
    print(boxes)
    box=boxes[0]
    print(box[0],box[1],box[0]-box[2],box[1]-box[3])

    # [379.06267792521123, 182.81498391636515, 74.38059083705849, 66.39324487334133]

    
    # Add second guy
    # y_index = 143 

        
    pass
    # create y_conf
    # 

    
if __name__ == "__main__":
    # iou_tests()
    # test_pre_process_image()
    # test_process_tiny_image_set()
    #test_batch_gen()
    # test_reconstruction()
    test_process_image_set("super_tiny")
    
    


    
# BOX =  [379.06267792521123, 182.81498391636515, 74.38059083705849, 66.39324487334133]
# dbox= 387.2 165.0 444.8 255.0 gt= 379.06267792521123 182.81498391636515 453.4432687622697 249.20822878970648
# intersection =  3824.2509047044623 Union= 6298.117876564173 IOUU= 0.6072053555769132
# 1 ) feat_map= [10, 8]  row= 6  col= 3  scale= [0, 0, 0.9, 1.5] y_loc_entry= [0.0043918983288272605, 0.06679562614484225, 0.25567246425718981, -0.30421435285833553]
# dbox= 400.0 174.0 457.6 222.0 gt= 379.06267792521123 182.81498391636515 453.4432687622697 249.20822878970648
# intersection =  2094.175346011559 Union= 5608.993435257074 IOUU= 0.37336027759418794
# 2 ) feat_map= [10, 8]  row= 6  col= 3  scale= [0.2, -0.2, 0.9, 0.8] y_loc_entry= [-0.21783032389339516, 0.3752417990215792, 0.25567246425718981, 0.32439430656403867]
# { y_loc non_zero: [568 569 570 573 574 575]
# { y_loc non_zero values: [ 0.0043919   0.06679562  0.25567245  0.37524179  0.25567245  0.32439432]
# For y_conf ... look at reconstruction_test.py
#n_matched: 2
