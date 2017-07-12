from model import BaseNet
import tensorflow as tf

class VGG16(BaseNet):
    def __init__(self,num_default_boxes,num_classes):
        super().__init__(num_default_boxes,num_classes)

    def graph(self,x,phase):
        """
        This will have two outputs - the coordinates for all default boxes and confidences
        """
        
        y_predict_box_coords = []
        y_predict_class = []
        
        # Convert from n,640,480 t0 300,300
        # resize the image
        # scipy.misc.resize(x,[x.shape[0],300,300,3])
        
        x = self.conv_layer_optional_pooling(x,64,(3,3),(1,1),"block1_conv1", phase, padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,64,(3,3),(1,1),"block1_conv2", phase, padding_type="SAME",pool_ksize=(2,2),pool_strides=(2,2),pool_name='block1_pool')
        
        x = self.conv_layer_optional_pooling(x,128,(3,3),(1,1),"block2_conv1", phase, padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,128,(3,3),(1,1),"block2_conv2", phase, padding_type="SAME",pool_ksize=(2,2),pool_strides=(2,2),pool_name='block2_pool')

        
        x = self.conv_layer_optional_pooling(x,256,(3,3),(1,1),"block3_conv1", phase, padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,256,(3,3),(1,1),"block3_conv2", phase, padding_type="SAME")
        
        
        
        x = self.conv_layer_optional_pooling(x,256,(3,3),(1,1),"block3_conv3", phase, padding_type="SAME",pool_ksize=(2,2),pool_strides=(2,2),pool_name='block3_pool')

        
        x = self.conv_layer_optional_pooling(x,512,(3,3),(1,1),"block4_conv1", phase, padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,512,(3,3),(1,1),"block4_conv2", phase, padding_type="SAME")
        

        
        x = self.conv_layer_optional_pooling(x,512,(3,3),(1,1),"block4_conv3", phase, padding_type="SAME",  pool_ksize=(2,2),pool_strides=(2,2),pool_name='block4_pool')

        x = self.conv_layer_optional_pooling(x,512,(3,3),(1,1),"block5_conv1", phase, padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,512,(3,3),(1,1),"block5_conv2", phase, padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,512,(3,3),(1,1),"block5_conv3", phase, padding_type="SAME")#  pool_ksize=(2,2),pool_strides=(2,2),pool_name='block5_pool')
        
        
        # LAYERS FROM SSD PAPER
        
        x = self.conv_layer_optional_pooling(x,1024,(3,3),(1,1),"block6_conv1", phase, padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,1024,(1,1),(1,1),"block7_conv1", phase, padding_type="SAME")
        
        print("\n   >>>>> FEATURE MAP 1:",x)
        self.convolve_and_collect(x,"block7_conv1",y_predict_box_coords,y_predict_class, phase )
        
        x = self.conv_layer_optional_pooling(x,256,(1,1),(1,1),"block8_conv1", phase, padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,512,(3,3),(2,2),"block8_conv2", phase, padding_type="SAME")
        
        print("\n   >>>>> FEATURE MAP 2:",x)
        self.convolve_and_collect(x,"block8_conv2",y_predict_box_coords,y_predict_class,phase)
        
        x = self.conv_layer_optional_pooling(x,128,(1,1),(1,1),"block9_conv1", phase, padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,256,(3,3),(2,2),"block9_conv2", phase, padding_type="SAME")
        
        print("\n   >>>>> FEATURE MAP 3:",x)
        self.convolve_and_collect(x,"block9_conv2",y_predict_box_coords,y_predict_class,phase)
        
        # Comment these out as pedestrians aren't going to be takeing up that much of the intput view. 
        x = self.conv_layer_optional_pooling(x,128,(1,1),(1,1),"block10_conv1", phase, padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,256,(3,3),(2,2),"block10_conv2", phase, padding_type="SAME")
        
        print("\n   >>>>> FEATURE MAP 4:",x)
        self.convolve_and_collect(x,"block10_conv2",y_predict_box_coords,y_predict_class,phase)
        
        print("Size OF BOX COORD PREDICT:",len(y_predict_box_coords))
        print("Size OF CLASSES CREDIT:",len(y_predict_class))
        
        y_predict_box_flat = tf.concat(y_predict_box_coords,1,name="y_predict_loc")
        y_predict_class_flat = tf.concat(y_predict_class,1,name="y_predict_conf")
        
        print("FLAT BOX COORD PREDICT:", y_predict_box_flat)
        print("FLAT OF CLASSES CREDIT:", y_predict_class_flat)
        
        # x = self.conv_layer_optional_pooling(x,128,(1,1),(1,1),"block11_conv1",padding_type="VALID")
        # x = self.conv_layer_optional_pooling(x,256,(3,3),(1,1),"block11_conv2",padding_type="VALID")
        # convolve_and_collect(x,y_predict_box_coords,y_predict_class)

        return [y_predict_box_flat,y_predict_class_flat]
    
    
