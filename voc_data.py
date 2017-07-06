import xml.etree.ElementTree as ET
import pickle

class VOCData:
    def __init__(self,source_data_dir):
        self.data_dir = source_data_dir
        self.global_lists = None

    def get_label_num(self,lbl):
        if lbl == "person":
            return 1
        return 0
        
    def get_train_test_splits(self):
        # VOC Data is already split into trainval
        # <annotation>
        # <filename>2011_003256.jpg</filename>
        # <folder>VOC2012</folder>
        # <object>
        #         <name>dog</name>
        #         <bndbox>
        #                 <xmax>398</xmax>
        #                 <xmin>246</xmin>
        #                 <ymax>218</ymax>
        #                 <ymin>130</ymin>
        #         </bndbox>

        if self.global_lists:
            return self.global_lists
        try:
            self.global_lists = pickle.load(open(self.source_data_dir + "/global_list.pkl","rb"))

        except:

            list_of_imgs = open( self.data_dir + "/ImageSets/Main/person_trainval.txt","r")
            train_list = {}
            test_list = {}

            for l in list_of_imgs:
                if l.find("-1") != -1:
                    continue

                # get the xml file
                parts = l.split(" ")
                tree  = ET.parse(self.data_dir + "/Annotations/"+parts[0]+".xml")
                root  = tree.getroot()
            
                # get the coordinates
                things = root.findall("./object")
                labels = []
                bboxes = []
                for i in things:
                    if i.find("name").text == "person":
                        box = i.find("bndbox")
                    
                        xmax = int(box.find("xmax").text)
                        ymax = int(box.find("ymax").text)                                        
                        xmin = int(box.find("xmin").text)                                        
                        ymin = int(box.find("ymin").text)
                        bbox = [xmin,ymin,xmax-xmin,ymax-ymin]
                        print(bbox)
                        bboxes.append(bbox)
                        labels.append("person")
                    
                fname = parts[0]+".jpg"        
                d = {"labels":labels,"bboxes":bboxes }
                numerical_parts = parts[0].split("_")
                if int(numerical_parts[1]) % 2:
                    test_list[fname] = d
                else:
                    train_list[fname]= d 

            self.global_lists = {"test":test_list, "train":train_list }
            pickle.dump(self.global_lists,open(self.data_dir + "/global_list.pkl","wb"))

        return self.global_lists

