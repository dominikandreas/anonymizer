import numpy as np
import tensorflow as tf

from anonymizer.utils import Box
from anonymizer.utils.helpers import get_default_session_config

class Detector:
    def __init__(self, kind, weights_path, gpu_memory_fraction: float = None):
        self.kind = kind
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(weights_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.session = tf.Session(graph=self.detection_graph, config=get_default_session_config(gpu_memory_fraction))

    def _convert_boxes(self, num_boxes, scores, boxes, image_height, image_width, detection_threshold):
        assert detection_threshold >= 0.001, 'Threshold can not be too close to "0".'

        result_boxes = []
        for i in range(int(num_boxes)):
            score = float(scores[i])
            if score < detection_threshold:
                continue

            y_min, x_min, y_max, x_max = map(float, boxes[i].tolist())
            box = Box(y_min=y_min * image_height, x_min=x_min * image_width,
                      y_max=y_max * image_height, x_max=x_max * image_width,
                      score=score, kind=self.kind)
            result_boxes.append(box)
        return result_boxes

    @property
    def image_tensor(self):
        return self.detection_graph.get_tensor_by_name('image_tensor:0')
    @property
    def num_detections(self):
        return self.detection_graph.get_tensor_by_name('num_detections:0')
    @property
    def detection_scores(self):
        return self.detection_graph.get_tensor_by_name('detection_scores:0')
    @property
    def detection_boxes(self):
        return self.detection_graph.get_tensor_by_name('detection_boxes:0')

    def detect_batch(self, images, detection_threshold):
        image_height, image_width, channels = images[0].shape
        assert channels == 3, f'Invalid number of channels: {channels}. ' \
                              f'Only images with three color channels are supported.'
        np_images = np.array(images)
        num_boxes_batch, scores_batch, boxes_batch = self.session.run(
            [self.num_detections, self.detection_scores, self.detection_boxes],
            feed_dict={self.image_tensor: np_images}
        )
        return [
            self._convert_boxes(num_boxes=num_boxes, scores=scores, boxes=boxes,
                                image_height=image_height, image_width=image_width,
                                detection_threshold=detection_threshold)
            for num_boxes, scores, boxes in zip(num_boxes_batch, scores_batch, boxes_batch)
        ]

    def detect(self, image, detection_threshold):
        return self.detect_batch([image], detection_threshold)[0]
