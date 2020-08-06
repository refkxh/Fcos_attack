import cv2
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
from coco_eval import COCOGenerator
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


def preprocess_img(image, input_ksize):
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side = input_ksize
    h, w, _ = image.shape

    smallest_side = min(w, h)
    largest_side = max(w, h)
    scale = min_side / smallest_side
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w = 32 - nw % 32
    pad_h = 32 - nh % 32

    image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
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
        nms_iou_threshold = 0.1
        max_detection_boxes_num = 100


    model = FCOSDetector(mode="inference", config=Config)
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./checkpoint/coco_37.2.pth", map_location=torch.device('cpu')))
    # model=convertSyncBNtoBN(model)
    # print("INFO===>success convert SyncBN to BN")
    model = model.eval()
    print("===>success loading model")

    import os

    root = r"../eval_code/select1000_new/"
    names = os.listdir(root)
    for name in names:
        img_bgr = cv2.imread(root + name)
        img_pad, nh, nw = preprocess_img(img_bgr, [800, 1333])
        img = cv2.cvtColor(img_pad.copy(), cv2.COLOR_BGR2RGB)
        img1 = transforms.ToTensor()(img)
        img1 = transforms.Normalize([0.40789654, 0.44719302, 0.47026115], [0.28863828, 0.27408164, 0.27809835], inplace=True)(img1)
        img1.unsqueeze_(dim=0)
        img1.requires_grad = True
        perturb = torch.zeros_like(img1)

        start_t = time.time()
        scores, classes, boxes = None, None, None
        for i in range(20):
            scores, classes, boxes = model(img1)
            loss = torch.sum(scores[0])
            if loss > 0:
                print(loss)
                model.zero_grad()
                loss.backward()
                grad = img1.grad.data.sign()
                img1 = img1 - 0.4 * grad
                img1 = torch.clamp(img1, -1, 1)
            else:
                break
        end_t = time.time()
        cost_t = 1000 * (end_t - start_t)
        print("===>success processing img, cost time %.2f ms" % cost_t)

        boxes = boxes[0].cpu().numpy().tolist()
        classes = classes[0].cpu().numpy().tolist()
        scores = scores[0].cpu().numpy().tolist()
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for i, box in enumerate(boxes):
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            img_pad = cv2.rectangle(img_pad, pt1, pt2, (0, 255, 0))
            b_color = colors[int(classes[i])]
            bbox = patches.Rectangle((box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], linewidth=1,
                                     facecolor='none', edgecolor=b_color)
            ax.add_patch(bbox)
            plt.text(box[0], box[1], s="%s %.3f" % (COCOGenerator.CLASSES_NAME[int(classes[i])], scores[i]), color='white',
                     verticalalignment='top',
                     bbox={'color': b_color, 'pad': 0})
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig('out_images/{}'.format(name), bbox_inches='tight', pad_inches=0.0)
        plt.close()
