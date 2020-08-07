import cv2
from model.fcos import FCOSDetector
import os
import torch
from torchvision import transforms
import numpy as np
from coco_eval import COCOGenerator
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODE = 1
attack_iters = 30
attack_epsilon = 0.05


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
        nms_iou_threshold = 0.2
        max_detection_boxes_num = 20


    model = FCOSDetector(mode="inference", config=Config)
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./checkpoint/coco_37.2.pth", map_location=torch.device('cpu')))
    # model=convertSyncBNtoBN(model)
    # print("INFO===>success convert SyncBN to BN")
    model = model.cuda().eval()
    print("===>success loading model")

    root = r"../eval_code/select1000_new/"
    names = os.listdir(root)
    for name in names:
        img_bgr = cv2.imread(root + name)
        mask = np.load('masks/' + name.split('.')[0] + '.npy')
        mask = np.expand_dims(mask, 2).repeat(3, axis=2).astype(np.uint8)
        img_pad, nh, nw = preprocess_img(img_bgr, [800, 1333])
        mask_resize, _, _ = preprocess_img(mask, [800, 1333])
        mask_resize = torch.from_numpy(mask_resize).cuda()
        mask_resize = mask_resize.permute(2, 0, 1)
        img = cv2.cvtColor(img_pad.copy(), cv2.COLOR_BGR2RGB)
        img1 = transforms.ToTensor()(img)
        img1 = transforms.Normalize([0.40789654, 0.44719302, 0.47026115], [0.28863828, 0.27408164, 0.27809835], inplace=True)(img1)
        img1.unsqueeze_(dim=0)
        img1 = img1.cuda()
        img1.requires_grad = True
        perturb = torch.zeros_like(img1).cuda()

        start_t = time.time()
        scores, classes, boxes = None, None, None
        for i in range(attack_iters):
            scores, classes, boxes = model(img1)
            loss = torch.sum(scores[0])
            if loss > 0:
                print('Iter {} loss:'.format(i), loss)
                model.zero_grad()
                loss.backward()
                grad = img1.grad.data.sign()
                img1.data = img1.data - grad * mask_resize * attack_epsilon
                img1.data = img1.data.clamp(-4, 4)
                perturb = perturb - grad * mask_resize * attack_epsilon
            else:
                break
        end_t = time.time()
        cost_t = 1000 * (end_t - start_t)
        print("===>success processing img, cost time %.2f ms" % cost_t)

        if MODE == 1:
            perturb.squeeze_(dim=0)
            perturb = perturb.cpu().numpy()
            perturb = perturb.transpose(1, 2, 0)
            perturb = perturb[:nh, :nw, :] * 0.28 * 255 + 128
            perturb = perturb.clip(0, 255)
            perturb = cv2.resize(perturb, (500, 500))
            perturb = (perturb - 128) * mask
            adv_img = img_bgr + cv2.cvtColor(perturb, cv2.COLOR_RGB2BGR)
            cv2.imwrite('out_images/{}'.format(name), adv_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        else:
            boxes = boxes[0].detach().cpu().numpy().tolist()
            classes = classes[0].detach().cpu().numpy().tolist()
            scores = scores[0].detach().cpu().numpy().tolist()
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
