import cv2
from model.fcos import FCOSDetector
import os
import torch
from torch import nn
from torchvision import transforms
import numpy as np
import time
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
attack_iters = 100
attack_epsilon = 0.01


def preprocess_img(image, input_ksize):
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side = input_ksize
    _, h, w = image.shape

    smallest_side = min(w, h)
    largest_side = max(w, h)
    scale = min_side / smallest_side
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    nw, nh = int(scale * w), int(scale * h)
    upsampling = nn.UpsamplingBilinear2d(size=(nh, nw))
    image = image.unsqueeze(dim=0)
    image_resized = upsampling(image)
    image_resized.squeeze_(dim=0)

    pad_w = 32 - nw % 32
    pad_h = 32 - nh % 32

    image_paded = torch.zeros(size=[3, nh + pad_h, nw + pad_w], dtype=torch.float)
    image_paded[:, :nh, :nw] = image_resized
    return image_paded, nh, nw


if __name__ == "__main__":
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)] * 4


    class Config:
        # backbone
        pretrained = False
        freeze_stage_1 = True
        freeze_bn = True

        # fpn
        fpn_out_channels = 256
        use_p5 = True

        # head
        class_num = 80
        use_GN_head = True
        prior = 0.01
        add_centerness = True
        cnt_on_reg = False

        # training
        strides = [8, 16, 32, 64, 128]
        limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

        # inference
        score_threshold = 0.3
        nms_iou_threshold = 0.2
        max_detection_boxes_num = 20


    model = FCOSDetector(mode="inference", config=Config)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./checkpoint/coco_37.2.pth", map_location=torch.device('cpu')))
    model = model.cuda().eval()
    print("===>success loading model")

    root = r"../eval_code/select1000_new/"
    names = os.listdir(root)
    for cnt, name in enumerate(names):
        img_bgr = cv2.imread(root + name)
        mask = np.load('masks/' + name.split('.')[0] + '.npy')
        mask = np.expand_dims(mask, 0).repeat(3, axis=0).astype(np.uint8)
        mask = torch.from_numpy(mask).cuda()
        img = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)
        img1 = transforms.ToTensor()(img).cuda()
        img1.requires_grad = True
        perturb = torch.zeros_like(img1).cuda()

        start_t = time.time()
        for i in range(attack_iters):
            img_pad, nh, nw = preprocess_img(img1, [800, 1333])
            img_pad = transforms.Normalize([0.40789654, 0.44719302, 0.47026115], [0.28863828, 0.27408164, 0.27809835],
                                           inplace=True)(img_pad)
            img_pad.unsqueeze_(dim=0)
            img_pad = img_pad.cuda()
            scores, classes, boxes = model(img_pad)
            loss = torch.sum(scores[0])
            if loss > 0:
                print('Iter {} loss:'.format(i), loss)
                model.zero_grad()
                loss.backward()
                grad = img1.grad.data.sign()
                img1.data = img1.data - grad * mask * attack_epsilon
                img1.data.clamp_(0, 1)
                perturb = perturb - grad * mask * attack_epsilon
            else:
                break
        end_t = time.time()
        cost_t = 1000 * (end_t - start_t)
        print("===>success processing img %d, cost time %.2f ms" % (cnt + 1, cost_t))

        perturb.squeeze_(dim=0)
        perturb = perturb.cpu().numpy()
        perturb = perturb.transpose(1, 2, 0)
        perturb = perturb * 255 + 128
        perturb = perturb.clip(0, 255)
        perturb = cv2.cvtColor(perturb, cv2.COLOR_RGB2BGR) - 128
        adv_img = (img_bgr + perturb).clip(0, 255)
        cv2.imwrite('out_images/{}'.format(name), adv_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
