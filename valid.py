from convnext import convnext_base
from trans import Transformer
import torchvision.transforms as transforms
import pandas as pd
import torch
import os
from PIL import Image

def valid(name):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = convnext_base(num_classes=2)
    model.head = torch.nn.Identity()

    model.eval()

    model.load_state_dict(torch.load('./extract_feature.pth'))
    model.to(device)

    net = Transformer()
    net.load_state_dict(torch.load('./transformer.pth'))

    net.to(device)

    times = ['1', '2', '3', '4', '5']
    visions = ['11.jpg', '12.jpg', '21.jpg', '22.jpg']
    event = ['Non_MACE','MACE']

    feature = torch.zeros((5, 4096)).to(device)
    feature_zero = torch.zeros((1, 1024)).to(device)
    i = 0

    for time in times:
        img_file = './img/' + name + '/' + time + '/'
        if os.path.exists(img_file):
            if os.path.exists(img_file + visions[0]):
                img1 = Image.open(img_file + visions[0]).convert('RGB')
                img1 = transform(img1).unsqueeze(0).to(device)
                feature1 = model(img1)
            else:
                feature1 = feature_zero

            if os.path.exists(img_file + visions[1]):
                img2 = Image.open(img_file + visions[1]).convert('RGB')
                img2 = transform(img2).unsqueeze(0).to(device)
                feature2 = model(img2)
            else:
                feature2 = feature_zero

            if os.path.exists(img_file + visions[2]):
                img3 = Image.open(img_file + visions[2]).convert('RGB')
                img3 = transform(img3).unsqueeze(0).to(device)
                feature3 = model(img3)
            else:
                feature3 = feature_zero

            if os.path.exists(img_file + visions[3]):
                img4 = Image.open(img_file + visions[3]).convert('RGB')
                img4 = transform(img4).unsqueeze(0).to(device)
                feature4 = model(img4)
            else:
                feature4 = feature_zero

            feature[i, :] = torch.cat([feature1, feature2, feature3, feature4], dim=1)
            i = i + 1



    output = net(feature.unsqueeze(0))

    pred = output.argmax(dim=1)


    print('The predict event is:', event[pred])
    print('The predict probalities is:', output[0].softmax(0)[1].item())

if __name__ == '__main__':
    valid('img')
