import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_np_image(image_path):
    image = Image.open(image_path).convert('RGB')
    np_image = np.array(image)
    return np_image


def save_np_image(image, image_path):
    pil_image = Image.fromarray((image).astype(np.uint8), mode='RGB')
    pil_image.save(image_path)


def save_detections(detections, detections_path):
    json_output = []
    for box in detections:
        json_output.append({
            'y_min': box.y_min,
            'x_min': box.x_min,
            'y_max': box.y_max,
            'x_max': box.x_max,
            'score': box.score,
            'kind': box.kind
        })
    with open(detections_path, 'w') as output_file:
        json.dump(json_output, output_file, indent=2)


class Anonymizer:
    def __init__(self, detectors, obfuscator, batch_size=8):
        self.detectors = detectors
        self.obfuscator = obfuscator
        self.batch_size = batch_size

    def anonymize_images_np(self, images, detection_thresholds):
        assert set(self.detectors.keys()) == set(detection_thresholds.keys()),\
            'Detector names must match detection threshold names'
        detected_boxes_batch = [[] for _ in images]
        for kind, detector in self.detectors.items():
            new_boxes_batch = detector.detect_batch(images, detection_threshold=detection_thresholds[kind])
            for batch_idx, new_boxes in enumerate(new_boxes_batch):
                detected_boxes_batch[batch_idx].extend(new_boxes)
        return [(self.obfuscator.obfuscate(image, detected_boxes), detected_boxes) 
                for image, detected_boxes in zip(images, detected_boxes_batch)]

    def anonymize_images(self, input_path, output_path, detection_thresholds, file_types, write_json):
        print(f'Anonymizing images in {input_path} and saving the anonymized images to {output_path}...')

        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        assert Path(output_path).is_dir(), 'Output path must be a directory'

        files = []
        for file_type in file_types:
            files.extend(list(Path(input_path).glob(f'**/*.{file_type}')))

        if len(files) > self.batch_size:
            batch_files = np.array(files).reshape(len(files) // self.batch_size, self.batch_size).tolist()
            batch_files.append(files[(len(files) + 1 // self.batch_size):])
        else:
            batch_files = [files]

        for path_batch in tqdm(batch_files):
            
            relative_paths = [img_path.relative_to(input_path) for img_path in path_batch]
            for relative_path in relative_paths:
                (Path(output_path) / relative_path.parent).mkdir(exist_ok=True, parents=True)

            output_image_paths = [output_path / relative_path for relative_path in relative_paths]
            output_detections_paths = [(output_path / relative_path).with_suffix('.json') for relative_path in relative_paths]

            # Anonymize image
            images = [load_np_image(str(input_image_path)) for input_image_path in path_batch]
            for idx, (anonymized_image, detections) in enumerate(
                self.anonymize_images_np(images, detection_thresholds)
            ):
                save_np_image(image=anonymized_image, image_path=str(output_image_paths[idx]))
                if write_json:
                    save_detections(detections=detections, detections_path=str(output_detections_paths[idx]))
