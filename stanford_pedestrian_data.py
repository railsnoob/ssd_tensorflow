import pickle

class StanfordPedestrianData:
    def __init__(self,source_data_dir):
        self.source_data_dir = source_data_dir
        self.global_lists = None


    def get_label_num(self,lbl):
        if lbl == "person":
            return 1
        if lbl == "people":
            return 1
        return 0

    def get_train_test_splits(self):
        if self.global_lists:
            return self.global_lists

        try:
            self.global_lists = pickle.load(open(self.source_data_dir + "/global_list.pkl","rb"))
        except:
            # file doesn't exist
            # Test list
            # Train
            test_list = {}
            train_list = {}
            annotations = pickle.load(open(self.source_data_dir + "/annotations.pkl","rb"))
            for a in annotations:
                for b in annotations[a]:
                    for frame_index in range(annotations[a][b]["nFrame"]):
                        frame_data = annotations[a][b]["frames"][frame_index]
                        labels = [ x['lbl'] for x in frame_data]
                        bboxes = [ x['pos'] for x in frame_data]
                        if not labels:
                            # discard empty frames
                            break
                        fname = "img{}{}{:04}.jpg".format(a,b,frame_index)
                        d = {"labels":labels,"bboxes":bboxes }
                        print(d)
                        if a in ['00','01','02','03','04','05','06']:
                            test_list[fname] = d
                        else:
                            train_list[fname]= d

            self.global_lists = { "test": test_list, "train": train_list }
            pickle.dump(global_lists, open(self.source_data_dir + "/global_list.pkl","wb"))

        return self.global_lists
