import cv2
import torch
import torchvision
import numpy as np
import PIL
import matplotlib.pyplot as plt

def predict(img):

    test_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(480),
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    img = test_transforms(img)
    imgs = img.unsqueeze(0)

    checkpoint = torch.load("C:\dev\Fingers\model_fingers_e7_l0.0040166958173116045.pth", map_location="cpu")
    model = torchvision.models.vgg11_bn(pretrained = False)
    model.classifier[-1] = torch.nn.Linear(model.classifier[3].out_features, 12)
    model.load_state_dict(checkpoint)
    model.eval()
    preds = torch.exp(torch.nn.functional.log_softmax(model(imgs), 1))
    torch.argmax(preds, 1)

    idx_to_class = {0: '0 - Esquerda',
                    1: '0 - Direita',
                    2: '1 - Esquerda',
                    3: '1 - Direita',
                    4: '2 - Esquerda',
                    5: '2 - Direita',
                    6: '3 - Esquerda',
                    7: '3 - Direita',
                    8: '4 - Esquerda',
                    9: '4 - Direita',
                    10: '5 - Esquerda',
                    11: '5 - Direita'
                    }

    prediction = idx_to_class[torch.argmax(preds[0]).item()]
    print("\nResultado:", prediction)
    return prediction

    #k = cv2.waitKey(30) & 0xff
    #if k == 27:
    #    break
 