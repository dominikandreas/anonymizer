import json
import time
import random
from pathlib import Path
import multiprocessing
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image
from progiter import ProgIter


def load_np_image(image_path) -> Optional[np.ndarray]:
    try:
        image = Image.open(image_path).convert('RGB')
        np_image = np.array(image)
        return np_image
        
    except Exception as e:
        print(f"Exception occurred while trying to load image {image_path}: {e}")
        return None


def save_np_image(image, image_path, compression_quality=-1):
    pil_image = Image.fromarray((image).astype(np.uint8), mode='RGB')
    if compression_quality != -1:
        image_path = str(Path(image_path).parent / (Path(image_path).stem + ".jpg"))
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


@dataclass
class QueueEntry:
    is_stop_entry: bool = False
    image_data: Optional[np.ndarray] = None
    output_paths: Optional[List[Path]] = None


batch_queue: "multiprocessing.Queue[QueueEntry]" = multiprocessing.Queue()


def get_next_batch(batch_size, img_path_gen, ignore_existing):
    is_finished = False
    batch_data: List[Tuple[np.ndarray, Path]] = []
    while len(batch_data) < batch_size:
        try:
            img_path, output_path = next(img_path_gen)
            if ignore_existing or not output_path.exists():
                img = load_np_image(img_path)
                if img is not None:
                    batch_data.append((img, output_path))
        except StopIteration:
            is_finished = True
    return batch_data, is_finished

    
def dataloader(batch_size: int,  input_output_paths: List[Tuple[Path, Path]], ignore_existing=True):
    img_path_gen = iter(input_output_paths)
    
    is_finished = False
    # load batches of images until the iterator has no remaining elements
    while True:
        batch_data, is_finished = get_next_batch(batch_size, img_path_gen, ignore_existing)
        
        # gather images of the same shape, as images of variying shapes cannot be inside the same batch
        while len(batch_data) > 0:
            next_batch_img_shape = batch_data[0][0].shape
            next_batch = [(img, out_data) for img, out_data in batch_data if img.shape == next_batch_img_shape]
            images, paths = zip(*next_batch)
            batch_queue.put(QueueEntry(image_data=images, output_paths=paths))
            batch_data = [(img, output_path) for img, output_path in batch_data if output_path not in paths]
            
        if is_finished:
             batch_queue.put(QueueEntry(is_stop_entry=True))
             break
         
        while batch_queue.qsize() > batch_size:
            time.sleep(.2)


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
        
    def start_dataloader(self, img_paths: List[Path], input_path, output_path, overwrite_images=False, reversed_processing=False):
        input_output_paths = [(img_path, output_path / img_path.relative_to(input_path))
                              for img_path in (reversed(img_paths) if reversed_processing else img_paths)]
        
        self.dataloader = multiprocessing.Process(
            target=dataloader, daemon=True,
            kwargs=dict(batch_size=self.batch_size, input_output_paths=input_output_paths,
                                           ignore_existing=not overwrite_images)
        )
        self.dataloader.start()

    def anonymize_images(self, input_path, output_path, detection_thresholds, file_types, write_json,
                         compression_quality=-1, overwrite_images=False, reversed_processing=False):
        print(f'Anonymizing images in {input_path} and saving the anonymized images to {output_path}...')

        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        assert output_path.is_dir(), 'Output path must be a directory'

        img_paths = []
        print("Gathering input image paths")
        for file_type in file_types:
            img_paths.extend(list(Path(input_path).glob(f'**/*.{file_type}')))

        print("Done. Starting data worker")
        self.start_dataloader(img_paths, input_path, output_path, 
                              overwrite_images=overwrite_images, reversed_processing=reversed_processing)        

        # use progiter instead of tqdm since it plays nicer with multiprocessing (tqdm isn't threadsafe?)
        progress_iter = iter(ProgIter(img_paths, desc="/".join(img_paths[0].parts[-4:-1])))
        while True:
            next_batch = batch_queue.get(timeout=10)
            if next_batch.is_stop_entry or len(next_batch.image_data) == 0:
                print("finished", "/".join(img_paths[0].parts[-4:-1]))
                break

            for output_image_path in next_batch.output_paths:
                output_image_path.parent.mkdir(exist_ok=True, parents=True)
                
            output_detections_paths = [output_path.with_suffix('.json') for output_path in  next_batch.output_paths]

            # Anonymize image
            anonymized_images_detections = self.anonymize_images_np(next_batch.image_data, detection_thresholds)
            for idx, (anonymized_image, detections) in enumerate(anonymized_images_detections):
                save_np_image(image=anonymized_image, image_path=str(next_batch.output_paths[idx]), compression_quality=compression_quality)
                if write_json:
                    save_detections(detections=detections, detections_path=str(output_detections_paths[idx]))
                    
            for _ in range(len(next_batch.output_paths)):
                next(progress_iter, None)
        
        self.dataloader.join()
