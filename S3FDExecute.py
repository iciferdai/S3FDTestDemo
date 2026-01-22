from S3FD import *
from Utils import *
import numpy as np

class S3FDExtractor:
    def __init__(self, model_path="./S3FD.pth", device="cuda"):
        self.model = S3FD(device=device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)  # 显式将模型移动到设备
        self.model.eval()

    def extract(self, input_image, is_bgr=True, is_resize=False, min_pixel_threshold=100,
                input_threshold=0.1, confidence_threshold=0.5, is_b_ext=True, max_num=100):

        if is_bgr: input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_scale=1.0
        if is_resize: input_image, input_scale = resize_img(input_image)

        input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float().to(self.model.device)
        with torch.no_grad():
            olist = self.model(input_tensor)
            # [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]
            logging.debug(f"Model infer result:\ncls1|{olist[0].shape}, reg1|{olist[1].shape}\n"
                          f"cls2|{olist[2].shape}, reg2|{olist[3].shape}\n"
                          f"cls3|{olist[4].shape}, reg3|{olist[5].shape}\n"
                          f"cls4|{olist[6].shape}, reg4|{olist[7].shape}\n"
                          f"cls5|{olist[8].shape}, reg5|{olist[9].shape}\n"
                          f"cls6|{olist[10].shape}, reg6|{olist[11].shape}")

        # 将置信度阈值传递给 refine 函数，并传递 min_pixel_threshold 和 is_resize 参数
        bboxlist = self.refine(olist, input_threshold, confidence_threshold, nms_thresh=0.3,
                               min_pixel_threshold=min_pixel_threshold, is_resize=is_resize,
                               input_scale=input_scale, is_b_ext=is_b_ext)

        logging.debug(f"detected_faces bboxlist length: {len(bboxlist)}")

        # max_num
        if len(bboxlist) > max_num:
            logging.info(f"detected_faces reduce from: {len(bboxlist)} -> {max_num}")
            bboxlist = bboxlist[:max_num]
        return bboxlist

    def refine(self, olist, input_thresh=0.1, confidence_thresh=0.5, nms_thresh=0.3, min_pixel_threshold=100,
               is_resize=False, input_scale=2.0, is_b_ext=True):
        bboxlist = []
        for i, (ocls, oreg) in enumerate(zip(olist[::2], olist[1::2])):
            logging.debug(f"i={i}, ocls|{ocls.shape}, oreg|{oreg.shape}")
            ocls = ocls.to("cpu")
            oreg = oreg.to("cpu")
            stride = 2 ** (i + 2)
            s_d2 = stride / 2
            s_m4 = stride * 4

            # 确保 ocls 和 oreg 的形状是 [channels, height, width]
            if len(ocls.shape) == 4: ocls = ocls.squeeze(0)
            if len(oreg.shape) == 4: oreg = oreg.squeeze(0)

            for hindex, windex in zip(*torch.where(ocls[1] > input_thresh)):
                score = ocls[1, hindex, windex].item()
                # pre-deal, faster
                if score < confidence_thresh:
                    continue

                loc = oreg[:, hindex, windex]
                priors = torch.tensor([windex * stride + s_d2, hindex * stride + s_d2, s_m4, s_m4])
                priors_2p = priors[2:]

                # 调整 priors 和 loc 的形状以匹配
                priors = priors.view(1, -1)
                loc = loc.view(1, -1)

                # 确保 loc 的形状在拼接维度上与 priors 一致
                box = torch.cat((
                    priors[:, :2] + loc[:, :2] * 0.1 * priors_2p.view(1, -1),
                    priors_2p.view(1, -1) * torch.exp(loc[:, 2:] * 0.2)
                ), dim=1)
                box = box.squeeze(0)

                box[:2] -= box[2:] / 2
                box[2:] += box[:2]

                # 提取坐标和置信度
                l, t, r, b = box[:]
                # 过滤掉尺寸小于 min_pixel_threshold 的人脸框
                if min((r - l), (b - t)) < min_pixel_threshold:
                    continue

                if is_resize:
                    l, t, r, b = [int(x * input_scale) for x in [l, t, r, b]] # 缩放坐标（如果需要）
                else:
                    l, t, r, b = int(l), int(t), int(r), int(b)
                if is_b_ext: b += (b - t) * 0.1

                bboxlist.append([l, t, r, b, score])

        if len(bboxlist) == 0:
            logging.info(f"refine bboxlist empty")
            return []

        logging.debug(f"refine init bboxlist len: {len(bboxlist)}")
        # 按置信度降序排序
        sorted_bbox = sorted(bboxlist, key=lambda x: x[-1], reverse=True)
        keep_id = self.refine_nms(sorted_bbox, nms_thresh)
        # 提取
        result = [sorted_bbox[idx] for idx in keep_id]
        return result

    def refine_nms(self, bboxlist, nms_thresh):
        dets = np.array(bboxlist, dtype=np.float32)
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            # 向量化计算IoU，无循环
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算重叠区域的宽高（避免负数）
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # 计算IoU
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # 保留IoU小于阈值的索引
            inds = np.where(ovr <= nms_thresh)[0]
            order = order[inds + 1]

        return keep