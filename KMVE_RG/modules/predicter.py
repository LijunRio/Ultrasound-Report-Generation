import os
from abc import abstractmethod
import cv2
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
from sentence_transformers import SentenceTransformer, util
from .tools import _prepare_device
from tqdm import tqdm

warnings.filterwarnings("ignore")
import sys

sys.path.append('../')


class BasePredictor(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args
        self.device, device_ids = _prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.metric_ftns = metric_ftns
        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.attn_pth = args.attn_pth

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.attn_pth): os.makedirs(self.attn_pth)
        self.sentence_bert = SentenceTransformer('distiluse-base-multilingual-cased')

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        self._train_epoch(1)


class Predictor(BasePredictor):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, test_dataloader):
        super(Predictor, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.test_dataloader = test_dataloader

    def calculate_metrics(self, pred, target, threshold=0.5):
        pred = pred.cpu()
        target = target.cpu()

        return {'precision': precision_score(y_true=target, y_pred=pred, average='micro'),
                'recall': recall_score(y_true=target, y_pred=pred, average='micro'),
                'f1': f1_score(y_true=target, y_pred=pred, average='micro'),
                }

    def _train_epoch(self, epoch):
        df = pd.DataFrame(columns=('key', 'gt', 'pred'))
        log = {}
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, cap_lens, reports_ids, reports_masks, mesh_label) in tqdm(enumerate(
                    self.test_dataloader), total=len(self.test_dataloader)):

                images, reports_ids, reports_masks, mesh_label = images.to(self.device), reports_ids.to(self.device), \
                                                                 reports_masks.to(self.device), mesh_label.to(
                    self.device)
                output, kmve_output, first_sentence_idx, first_attmap, first_sentence_probs = self.model(images,
                                                                                                         mode='evaluate')

                cur_pth = os.path.join(self.attn_pth, str(batch_idx))
                if not os.path.exists(cur_pth): os.makedirs(cur_pth)
                sentence_txt = self.model.tokenizer.decode(first_sentence_idx)

                res_txt = os.path.join(cur_pth, 'result_sentence.txt')
                with open(res_txt, 'w') as file:
                    file.write(sentence_txt)

                first_stentence = self.model.tokenizer.decode_list(first_sentence_idx)

                image1 = images[:, 0, :][0].cpu().numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image1 = image1 * std + mean
                image1 = np.clip(image1, 0, 1)

                image_data1 = image1 / image1.max()
                image_data1 = image_data1 * 255
                img1 = image_data1.astype('uint8')
                probs_list = []
                for i in range(len(first_attmap)):
                    cur_word = first_stentence[i]
                    cur_prob = float(first_sentence_probs[i].cpu())
                    probs_list.append(cur_prob)
                    cur_map = first_attmap[i]
                    cur_list = torch.split(cur_map, 49)
                    cur_map_0, cur_map_1 = cur_list[0], cur_list[1]
                    cam_0 = np.array(cur_map_0.permute(1, 0).reshape(7, 7).cpu())
                    cam_0 = cam_0 - np.min(cam_0)
                    cam_img0 = cam_0 / np.max(cam_0)
                    cam_img0 = np.uint8(255 * cam_img0)

                    cam_img0 = cv2.resize(cam_img0, (224, 224))
                    heatmap0 = cv2.applyColorMap(cam_img0, cv2.COLORMAP_JET)
                    dst = cv2.addWeighted(img1, 0.7, heatmap0, 0.3, 0)
                    filename = os.path.join(cur_pth, str(i) + '-' + cur_word + '-' + str(round(cur_prob, 2)) + '.jpg')
                    filename = str(filename).replace('<start/end>', 'se')
                    cv2.imencode('.jpg', dst)[1].tofile(filename)
                attn_csv = pd.DataFrame({'word': first_stentence, 'prob': probs_list})
                attn_csv.to_csv(os.path.join(cur_pth, 'attn_result.csv'), index=False)

                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                for i in range(len(reports)):
                    pre = reports[i]
                    gt = ground_truths[i]
                    df = pd.concat([df, pd.Series({'key': images_id[i], 'gt': gt, 'pred': pre})], ignore_index=True)
                test_res.extend(reports)
                test_gts.extend(ground_truths)

        self.lr_scheduler.step()
