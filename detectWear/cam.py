import os
import argparse
import cv2
import torch
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import model.model as module_arch

from utils import get_testloader, get_attentionloader

def create_cam(config):
    if not os.path.exists(config.result_path):
        os.mkdir(config.result_path)

    # 데이터 로드
    test_loader, num_class = get_testloader(config.dataset,
                                                  config.dataset_path,
                                                  config.img_size)
    attention_val_data_loader, _ = get_attentionloader(config.dataset,
                                            './dataset/final_mask/val/',
                                            7)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    cnn = module_arch.resnet101()
    num_ftrs = cnn.fc.in_features
    cnn.fc = nn.Linear(num_ftrs+1, 3)

    checkpoint = torch.load(os.path.join(config.model_path, config.model_name))
    state_dict = checkpoint['state_dict']
    cnn.load_state_dict(state_dict)
    cnn=cnn.to(device)


    # hook
    feature_blobs = []

    def hook_feature(module, input, output):
        feature_blobs.append(input[0].cpu().data.numpy())

    # 마지막 컨볼루션 레이어 가져오기
    cnn._modules['avgpool'].register_forward_hook(hook_feature)
    params = list(cnn.parameters())
    # get weight only from the last layer(linear)
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())


    def returnCAM(feature_conv, weight_softmax, class_idx):
        size_upsample = (config.img_size, config.img_size)
        _, nc, h, w = feature_conv.shape
        output_cam = []
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    # 이미지에 적용하기
    cnn.eval()
    classtire = ['danger', 'safety', 'warning']
    for i, (val, attention) in enumerate(zip(test_loader, attention_val_data_loader)):
        image_tensor, label = val[0].to(device), val[1].to(device)
        attention = attention[0].to(device)

        # 이미지 불러와서 result 폴더에 저장
        image_PIL = transforms.ToPILImage()(image_tensor[0])
        image_PIL.save(os.path.join(config.result_path, 'img%d.png' % (i + 1)))

        image_tensor = image_tensor.to(device)
        label = label.to(device)

        logit = cnn(image_tensor, attention)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        print("True label : %d, Predicted label : %d, Probability : %.2f" % (label.item(), idx[0].item(), probs[0].item()))

        CAMs = returnCAM(feature_blobs[0], weight_softmax, [label.item()])
        img = cv2.imread(os.path.join(config.result_path, 'img%d.png' % (i + 1)))
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        # 불러온 이미지에 CAM 결과를 적용해서 다시 저장
        cv2.imwrite(os.path.join(config.result_path, 'cam'+str(i + 1)                       # num
                                 +'_gt('+str(classtire[label.item()])                       # gt
                                 +')_pred('+str(classtire[idx[0].item()])                   # pred
                                 +str(round(probs[0].item(),2))+').png'), result)           # probs

        if i + 1 == config.num_result:
            break
        feature_blobs.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='warning', choices='OWN')
    parser.add_argument('--dataset_path', type=str, default='./dataset/data_3class/val/')
    parser.add_argument('--model_path', type=str, default='./saved/models/detectTireWear/0720_053829/')
    parser.add_argument('--model_name', type=str, default='model_best.pth')

    parser.add_argument('--result_path', type=str, default='./saved/cam/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_result', type=int, default=108)

    config = parser.parse_args()
    print(config)

    create_cam(config)