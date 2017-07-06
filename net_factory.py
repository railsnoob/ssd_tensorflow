from vgg16 import VGG16
from alex_net import AlexNet

class NetFactory:
    def __init__(self):
        pass

    def get_net(net_name):
        if net_name == "vgg16":
            return VGG16
        elif net_name == "alexnet":
            return AlexNet
