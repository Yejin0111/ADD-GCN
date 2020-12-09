import torchvision
from .add_gcn import ADD_GCN

model_dict = {'ADD_GCN': ADD_GCN}

def get_model(num_classes, args):
    res101 = torchvision.models.resnet101(pretrained=True)
    model = model_dict[args.model_name](res101, num_classes)
    return model