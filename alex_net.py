from model import BaseNet
import tensorflow as tf

class AlexNet(BaseNet):
    def __init__(self,num_default_boxes,num_classes):
        super().__init__(num_default_boxes,num_classes)
    
    def graph(self,x):
        """
        Alex NET!
        """
        y_predict_box_coords = []
        y_predict_class = [] 

        x= self.conv_layer_optional_pooling(x,96,(11,11),(4,4),"conv1",padding_type= "SAME",pool_ksize=(3,3),pool_strides=(2,2),pool_name="pool1")
        
        x= self.conv_layer_optional_pooling(x,192,(5,5),(1,1),"conv2",padding_type= "SAME",pool_ksize=(3,3),pool_strides=(2,2),pool_name="pool2")

        x= self.conv_layer_optional_pooling(x,384,(3,3),(1,1),"conv3")
        x= self.conv_layer_optional_pooling(x,384,(3,3),(1,1),"conv4")
        x= self.conv_layer_optional_pooling(x,256,(3,3),(1,1),"conv5",padding_type= "SAME",pool_ksize=(3,3),pool_strides=(2,2),pool_name="pool5")

        # Additional SSD Layers

        # LAYERS FROM SSD PAPER
        
        x = self.conv_layer_optional_pooling(x,1024,(3,3),(1,1),"conv6",padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,1024,(1,1),(1,1),"conv7",padding_type="SAME")
        
        print("\n   >>>>> FEATURE MAP 1:",x)
        self.convolve_and_collect(x,"conv7_collect",y_predict_box_coords,y_predict_class)
        
        x = self.conv_layer_optional_pooling(x,256,(1,1),(1,1),"conv8",padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,512,(3,3),(2,2),"conv9",padding_type="SAME")
        
        print("\n   >>>>> FEATURE MAP 2:",x)
        self.convolve_and_collect(x,"conv9_collect",y_predict_box_coords,y_predict_class)
        
        x = self.conv_layer_optional_pooling(x,128,(1,1),(1,1),"conv10",padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,256,(3,3),(2,2),"conv11",padding_type="SAME")
        
        print("\n   >>>>> FEATURE MAP 3:",x)
        self.convolve_and_collect(x,"conv11_collect",y_predict_box_coords,y_predict_class)

        # Comment these out as pedestrians aren't going to be takeing up that much of the intput view. 
        x = self.conv_layer_optional_pooling(x,128,(1,1),(1,1),"conv12",padding_type="SAME")
        x = self.conv_layer_optional_pooling(x,256,(3,3),(2,2),"conv13",padding_type="SAME")
        
        print("\n   >>>>> FEATURE MAP 4:",x)
        self.convolve_and_collect(x,"conv13_collect",y_predict_box_coords,y_predict_class)

        
        print("ALEX Size OF BOX COORD PREDICT:",len(y_predict_box_coords))
        print("ALEX Size OF CLASSES CREDIT:",len(y_predict_class))
        
        y_predict_box_flat = tf.concat(y_predict_box_coords,1)
        y_predict_class_flat = tf.concat(y_predict_class,1)
        
        print("ALEX FLAT BOX COORD PREDICT:", y_predict_box_flat)
        print("ALEX FLAT OF CLASSES CREDIT:", y_predict_class_flat)
        
        return [y_predict_box_flat,y_predict_class_flat]
