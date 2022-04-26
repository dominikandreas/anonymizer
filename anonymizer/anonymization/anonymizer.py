import os
import sys
import json
import atexit
import logging
from queue import Empty
import time
from pathlib import Path
import multiprocessing
from typing import List, Tuple, Optional, Iterator, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from PIL import Image
from progiter import ProgIter


if TYPE_CHECKING:
    from anonymizer.detection.detector import Detector
    from anonymizer.obfuscation.obfuscator import Obfuscator


def load_np_image(image_path) -> Optional[np.ndarray]:
    try:
        image = Image.open(image_path).convert('RGB')
        np_image = np.array(image)
        return np_image
        
    except Exception as e:
        print(f"Exception occurred while trying to load image {image_path}: {e}", file=sys.stderr)
        return None


def save_np_image(image, image_path: str, compression_quality=-1):
    pil_image = Image.fromarray((image).astype(np.uint8), mode='RGB')
    if compression_quality != -1:
        image_path = str(Path(image_path).with_suffix(".jpg"))
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


def get_next_batch(batch_size, img_path_gen, load_processed):
    is_finished = False
     # contains tuples of images and target output path. images are only loaded conditionally and may be None
    batch_data: List[Tuple[Optional[np.ndarray], Path]] = [] 
    while len(batch_data) < batch_size:
        try:
            img_path, output_path = next(img_path_gen)
            img = load_np_image(img_path) if (not output_path.exists() or load_processed) else None
            batch_data.append((img, output_path))
        except StopIteration:
            is_finished = True
            break
            
    
    for output_path in {p.parent for _, p in batch_data}:
        output_path.parent.mkdir(exist_ok=True, parents=True)
    return batch_data, is_finished


def filter_batch(batch_data):
    """Return a subset of the given batch data containing only consistent images (same shape or None)."""
    next_img = batch_data[0][0]
    next_batch = [(img, path) for img, path in batch_data 
                  if (next_img is None and img is None) or
                  (img is not None and next_img is not None and img.shape == next_img.shape)]
    paths = {p for _, p in next_batch}
    remaining = [(img, output_path) for img, output_path in batch_data if output_path not in paths]
    return next_batch, remaining
    
    
def dataloader(batch_size: int,  input_output_path_iterator: Iterator[Tuple[Path, Path]], 
               ignore_existing=True, loop_until_complete=True):
    """Load batches of images and put them into the queue until the iterator has no remaining elements."""
    
    while True:
        batch_data, is_finished = get_next_batch(batch_size, input_output_path_iterator, ignore_existing)
        
        # Gather images of the same shape, as images of variying shapes cannot be inside the same batch
        while len(batch_data) > 0:
            next_batch, batch_data = filter_batch(batch_data)
            images, paths = zip(*next_batch)
            batch_queue.put(QueueEntry(image_data=images, output_paths=paths))
            
        if is_finished:
            batch_queue.put(QueueEntry(is_stop_entry=True))
            return
         
        if not loop_until_complete:
            return
        else:
            # Wait until previous batches have been consumed before loading further images
            while batch_queue.qsize() > 2:
                time.sleep(.01)
        
   

@dataclass
class Anonymizer:
    detectors: List["Detector"]
    """Detectors to apply."""
    obfuscator: List["Obfuscator"]
    """Obfuscators to apply (for blurring detection regions)."""
    
    batch_size: int = 8
    """Number of images to process in a single forward pass (increases GPU memory consuption)."""
    parallel_dataloading: bool = True
    """Whether to start a parallel process for dataloading."""
    reversed_processing: bool = False
    """Whether to process images in reverse."""
    overwrite_existing: bool = False
    """Whether to process input images where the output already exists."""
    compression_quality: int = -1
    """Controls the jpeg compression. Set to -1 to disable, or value between 0 (max compression) and 100 (lossless)."""
    save_detection_json_files: bool = False
    """Whether to store a json file for each image containing the detected bounding boxes."""
    
    _img_paths_: Optional[List[Path]] = None

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

    def get_next_batch(self, timeout_in_s=300) -> QueueEntry:
        if not self.parallel_dataloading:
            # Manually generate the next element in the queue
            dataloader(batch_size=self.batch_size, input_output_path_iterator=self.data_iter,
                       ignore_existing=self.overwrite_existing)
        for i in range(10):
            try:
                return batch_queue.get(timeout=timeout_in_s / 10)
            except Empty:
                print(f"Dataloader timeout {i+1}/10, waiting another {timeout_in_s / 10:.2f}s "
                        "for next batch...", file=sys.stderr)
                sys.stderr.flush()
        print(f"Unable to get any data within a timeout of {timeout_in_s}", file=sys.stderr)
        sys.stderr.flush()
        return QueueEntry(is_stop_entry=True)

            
    def prepare_dataloading(self, img_paths: List[Path], input_path, output_path):
        input_output_paths = [(img_path, output_path / img_path.relative_to(input_path))
                              for img_path in (reversed(img_paths) if self.reversed_processing else img_paths)]
        
        if self.compression_quality != -1:
            # Set output file type to jpg if compression is active
            input_output_paths = [(img, path.with_suffix(".jpg")) for img, path in input_output_paths]
            
        self.data_iter = iter(input_output_paths)
        
        if self.parallel_dataloading:
            print("Starting data worker")
            self.dataloader = multiprocessing.Process(
                target=dataloader, daemon=True,
                kwargs=dict(batch_size=self.batch_size, input_output_path_iterator=self.data_iter,
                            ignore_existing=self.overwrite_existing)
            )
            atexit.register(lambda: self.dataloader.terminate())
            self.dataloader.start()
              
    def prepare_anonymization(self, input_path, file_types, output_path):
        print(f'Anonymizing images in {input_path} and saving the anonymized images to {output_path}...')

        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        assert output_path.is_dir(), 'Output path must be a directory'

        if self._img_paths_ is None:
            self._img_paths_ = []
            print("Gathering input image paths... ", end="", file=sys.stderr)
            sys.stderr.flush()
            for file_type in file_types:
                self._img_paths_.extend(list(Path(input_path).glob(f'**/*.{file_type}')))

        print("Done. ", file=sys.stderr)
        self.prepare_dataloading(self._img_paths_, input_path, output_path)        


    def process_batch(self, batch, detection_thresholds):
        output_detections_paths = [output_path.with_suffix('.json') for output_path in  batch.output_paths]

        if batch.image_data[0] is not None:  # image data is set to None for batches that don't require processing
            # Anonymize image
            anonymized_images_detections = self.anonymize_images_np(batch.image_data, detection_thresholds)
            
            for idx, (anonymized_image, detections) in enumerate(anonymized_images_detections):
                
                save_np_image(image=anonymized_image, image_path=str(batch.output_paths[idx]), 
                            compression_quality=self.compression_quality)
                
                if self.save_detection_json_files:
                    save_detections(detections=detections, detections_path=str(output_detections_paths[idx]))
                    
    def anonymize_images(self, input_path, output_path, detection_thresholds, file_types):
        self.prepare_anonymization(input_path, file_types, output_path)

        # use progiter instead of tqdm since it plays nicer with multiprocessing (tqdm apparently isn't threadsafe)
        progress = ProgIter(self._img_paths_, desc="/".join(self._img_paths_[0].parts[-4:-1]), stream=sys.stderr)
        last_percent = -1
        progress_iter = iter(progress)
        while True:
            try:
                next_batch = self.get_next_batch()
                if next_batch.is_stop_entry or len(next_batch.image_data) == 0:
                    print("\nfinished", "/".join(self._img_paths_[0].parts[-4:-1]), file=sys.stderr)
                    return
                    
                self.process_batch(next_batch, detection_thresholds)
                        
                # Iterate over the progress iterator to log the progress
                for _ in range(len(next_batch.output_paths)):
                    next(progress_iter, None)
                
                sys.stderr.flush()
                
                percent = int(progress._now_idx / progress.total * 100)
                if percent != last_percent:
                    logging.info(progress.format_message())
                
            except Exception as e:
                logging.exception(str(e))
                exit(-1)
            