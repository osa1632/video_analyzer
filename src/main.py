import json
import pathlib
import sys
import typing

import cv2
import numpy as np
import torch
from torchvision import transforms

sys.path.append('./src')
from src.FROM.lib.models.fpn import LResNet50E_IR_Occ

# from retinaface import RetinaFace
from src.Pytorch_Retinaface.models.retinaface import RetinaFace
from src.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from src.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from src.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms


class NumpyEncoder(json.JSONEncoder):
    # Numpy np.array/types are not serializable:
    # see: https://bobbyhadz.com/blog/python-typeerror-object-of-type-int32-is-not-json-serializable
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ProcessingStage(object):
    def __init__(self, **kwargs):
        raise NotImplementedError(f"{self.__class__} is only a base class")
    def run(self, frame, kwargs):
        return

    def close(self):
        return


class FaceDetection(ProcessingStage):
    def __init__(self, confidence_threshold=0.9, nms_threshold=0.1, **kwargs):
        self.cfg_mnet = {
            'name': 'mobilenet0.25',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': True,
            'batch_size': 32,
            'ngpu': 1,
            'epoch': 250,
            'decay1': 190,
            'decay2': 220,
            'image_size': 640,
            'pretrain': False,
            'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
            'in_channel': 32,
            'out_channel': 64
        }

        def load_model(model, pretrained_path, load_to_cpu):
            print('Loading pretrained model from {}'.format(pretrained_path))
            if load_to_cpu:
                pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
            else:
                device = torch.cuda.current_device()
                pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
            if "state_dict" in pretrained_dict.keys():
                pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
            else:
                pretrained_dict = remove_prefix(pretrained_dict, 'module.')
            model.load_state_dict(pretrained_dict, strict=False)
            return model

        def remove_prefix(state_dict, prefix):
            ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
            print('remove prefix \'{}\''.format(prefix))
            f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
            return {f(key): value for key, value in state_dict.items()}

        self.detector = RetinaFace(cfg=self.cfg_mnet, phase='test')
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        load_model(self.detector, './weights/mobilenet0.25_Final.pth', load_to_cpu=True)
        self.detector.eval()

    def _preprocess(self, img_raw):
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)

        return img

    def process(self, frame):
        priorbox = PriorBox(self.cfg_mnet, image_size=(frame.shape[0], frame.shape[1]))
        img = self._preprocess(frame)
        ret = self.detector(img)
        priors = priorbox.forward()
        prior_data = priors.data
        scale = torch.Tensor([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])

        loc, conf, landms = ret

        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg_mnet['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg_mnet['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        # order = scores.argsort()[::-1][:args.top_k]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        # dets = np.concatenate((dets, landms), axis=1)
        return {'facial_area': dets, 'kps': landms}

    def run(self, frame, kwargs):
        kwargs['detection_resuls'] = self.process(frame)


class TrackFace(ProcessingStage):
    def __init__(self, **kwargs):
        from src.sort import Sort
        self.tracker = Sort()

    def run(self, frame, kwargs):
        try:
            dets = kwargs['detection_resuls']['facial_area']
            trackers = self.tracker.update(dets)
            kwargs['trackers'] = trackers
        except Exception as e:
            print(e)
            kwargs['trackers'] = []


class ScoreFace(ProcessingStage):
    def __init__(self, **kwargs):
        self.net = LResNet50E_IR_Occ(num_mask=226).cpu()
        checkpoint = torch.load('weights/model_p5_w1_9938_9470_6503.pth.tar', map_location='cpu')
        state_dict = checkpoint['state_dict']

        if isinstance(self.net, torch.nn.DataParallel):
            self.net.module.load_state_dict(state_dict, strict=False)
        else:
            self.net.load_state_dict(state_dict, strict=False)
        self.net.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])

    def run(self, frame, kwargs):
        kwargs['score'] = []
        for bbox in kwargs['trackers']:
            bbox_int = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            crop_frame = cv2.resize(frame[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2], ...], (96, 112),
                                    cv2.INTER_LINEAR)
            fc_mask, mask, vec, fc = self.net(self.transform(crop_frame).unsqueeze(0), None)

            fc_mask_np, fc_np = fc_mask.detach().numpy(), fc.detach().numpy()
            score = float(1 - np.sum(fc_mask_np * fc_np) / (np.linalg.norm(fc_np) * np.linalg.norm(fc_mask_np)))
            kwargs['score'].append((bbox_int, score))


class FrameIdTracking(ProcessingStage):
    def __init__(self, **kwargs):
        self.frame_id = 0

    def run(self, frame, kwargs):
        kwargs['frame_id'] = self.frame_id
        print(self.frame_id)
        self.frame_id += 1


class ScoreSaver(ProcessingStage):
    def __init__(self, input_name=None, **kwargs):
        input_name = input_name or ''
        self.json_file = open(f'{pathlib.Path(input_name).stem}.json', 'a')

    def run(self, frame, kwargs):
        json_str = json.dumps({kwargs.get('frame_id', -1): kwargs.get('score', -1)}) + '\n'
        self.json_file.write(json_str)


class BlurFace(ProcessingStage):
    def __init__(self, **kwargs):
        ...

    def run(self, frame, kwargs):
        mask = np.zeros_like(frame)
        blur_frame = cv2.blur(frame, ksize=(20, 20))
        for face in kwargs['trackers']:
            x, y, x2, y2, _ = face.astype(int)
            mask = cv2.rectangle(mask, (x, y), (x2, y2), color=(1, 1, 1), thickness=-1)
            # frame = cv2.rectangle(frame, (x, y), (x2, y2), color=(255, 0, 0), thickness=5)
            # blur_frame = cv2.rectangle(blur_frame, (x, y), (x2, y2), color=(255, 0, 0), thickness=5)
        kwargs['out_frame'] = (frame * (1 - mask) + blur_frame * mask).astype('uint8')

        # for bbox_score in kwargs['score']:
        #     cv2.putText(frame, f"{bbox_score[1]}", bbox_score[0][:2], cv2.FONT_HERSHEY_DUPLEX,
        #                 0.5, (255, 255, 255))
        # plt.imshow(frame[...,::-1])
        # plt.show()


class OutputVideoSaver(ProcessingStage):
    def __init__(self, input_name, **kwargs):
        input_path = pathlib.Path(input_name)
        self.output_path = f'{input_path.stem}_output.mp4'
        self.output_video = None
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    def run(self, frame, kwargs):
        if self.output_video is None:
            self.output_video = cv2.VideoWriter(self.output_path, self.fourcc, 20.,
                                                frame.shape[:-1][::-1])
        self.output_video.write(kwargs['out_frame'])

    def close(self):
        if self.output_video is not None:
            self.output_video.release()
            cv2.destroyAllWindows()


class VideoAnalytic(object):
    def __init__(self, input_name, confidence_threshold = 0.9, nms_threshold=0.1):
        self.input_name = input_name
        self.input_path = pathlib.Path(input_name)

        self.input_stream = None
        self.output_stream = None
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.processing_stages: typing.List[ProcessingStage] = []

        self.set_stream()

    def add_processing_stage(self, processing_stage):
        self.processing_stages.append(processing_stage(**self.__dict__))

    def close_processing_stages(self):
        for processing_stage in self.processing_stages:
            processing_stage.close()

    def set_stream(self):
        self.input_stream = cv2.VideoCapture(self.input_name)

    def process_stream(self, frame, kwargs=None):
        kwargs = kwargs or {}
        for processing_stage in self.processing_stages:
            processing_stage.run(frame, kwargs)

    def analyze_video(self):
        ret = True
        # all_results = []
        frame_id = 0
        ret, frame = self.input_stream.read()
        while ret and frame_id < 30:
            self.process_stream(frame)
            ret, frame = self.input_stream.read()
            frame_id += 1
        self.close_processing_stages()

class BlurStage(ProcessingStage):
    def __init__(self, **kwargs):
        self.blur_face = BlurFace(**kwargs)
        self.video_saver = OutputVideoSaver(**kwargs)

    def run(self, frame, kwargs):
        self.blur_face.run(frame, kwargs)
        self.video_saver.run(frame, kwargs)

    def close(self):
        self.video_saver.close()

class DetectTrackScore(ProcessingStage):
    def __init__(self, **kwargs):
        self.detect_stage = FaceDetection(**kwargs)
        self.track_stage = TrackFace(**kwargs)
        self.score_stage = ScoreStage(**kwargs)

    def run(self, frame, kwargs):
        self.detect_stage.run(frame, kwargs)
        self.track_stage.run(frame, kwargs)
        self.score_stage.run(frame, kwargs)

class ScoreStage(ProcessingStage):
    def __init__(self, **kwargs):
        self.score_face = ScoreFace(**kwargs)
        self.score_saver = ScoreSaver(**kwargs)

    def run(self, frame, kwargs):
        self.score_face.run(frame, kwargs)
        self.score_saver.run(frame, kwargs)

if __name__ == '__main__':
    input_name = sys.argv[1]
    confidence_threshold = float(sys.argv[2])
    video_analytic = VideoAnalytic(input_name=input_name, confidence_threshold=confidence_threshold)
    # video_analytic.add_processing_stage(FrameIdTracking)
    video_analytic.add_processing_stage(DetectTrackScore)
    video_analytic.add_processing_stage(BlurStage)
    video_analytic.analyze_video()
