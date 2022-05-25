import os
import sys
import json
import atexit
import logging
import random
from queue import Empty
import time
from pathlib import Path
import multiprocessing
from typing import List, Tuple, Optional, Iterator, TYPE_CHECKING, Dict
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
        logging.info(f"Exception occurred while trying to load image {image_path}: {e}", file=sys.stderr)
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
    failed: bool = False
    image_data: Optional[np.ndarray] = None
    output_paths: Optional[List[Path]] = None
    
    
@dataclass
class OutputJob:
    is_stop_entry: bool = False
    image_data: Optional[np.ndarray] = None
    output_path: Optional[Path] = None
    compression_quality: Optional[int] = None
    detections: Optional[Dict] = None
    
    def execute(self):
        if self.is_stop_entry:
            return
        if self.image_data is not None:
            save_np_image(self.image_data, self.output_path, self.compression_quality)
        if self.detections and self.detections is not None:
            save_detections(detections=self.detections, detections_path=self.output_path.with_suffix('.json'))
    

batch_queue: "multiprocessing.Queue[QueueEntry]" = multiprocessing.Queue()
output_queue: "multiprocessing.Queue[OutputJob]" = multiprocessing.Queue(maxsize=128)
completed_queue: "multiprocessing.Queue[QueueEntry]" = multiprocessing.Queue()


def get_next_batch_data(batch_size, img_path_gen, load_processed):
    is_finished = False
     # contains tuples of images and target output path. images are only loaded conditionally and may be None
    batch_data: List[Tuple[Optional[np.ndarray], Path]] = [] 
    num_skipped = 0
    while len(batch_data) < batch_size:
        try:
            img_path, output_path = next(img_path_gen)
            if not img_path.exists():
                logging.warning(f"input path {img_path} does not exist!")
            if output_path.exists():
                num_skipped += 1
                
            img = load_np_image(img_path) if (not output_path.exists() or load_processed) else None
            batch_data.append((img, output_path))
        except StopIteration:
            is_finished = True
            break
            
    if num_skipped > 0:
        logging.debug(f"Skipped {num_skipped}/{len(batch_data)} images because anonymized versions exist.")
        
    for output_path in {p.parent for _, p in batch_data}:
        output_path.parent.mkdir(exist_ok=True, parents=True)
    return batch_data, is_finished


def filter_batch(batch_data, max_size: int):
    """Return a subset of the given batch data containing only consistent images (same shape or None)."""
    next_img = batch_data[0][0]
    next_batch = [(img, path) for img, path in batch_data 
                  if (next_img is None and img is None) or
                  (img is not None and next_img is not None and img.shape == next_img.shape)]
    next_batch = next_batch[:max_size]
    paths = {p for _, p in next_batch}
    
    remaining = [(img, output_path) for img, output_path in batch_data if output_path not in paths]
    return next_batch, remaining
    
    
def dataloader(batch_size: int,  input_output_path_iterator: Iterator[Tuple[Path, Path]], 
               ignore_existing=True, loop_until_complete=True):
    """Load batches of images and put them into the queue until the iterator has no remaining elements."""
    try:
        batch_data = []
        while True:
            logging.debug(f"{os.getpid()} getting next batch...")
            next_batch_data, is_finished = get_next_batch_data(batch_size * 4, input_output_path_iterator, ignore_existing)
            batch_data += next_batch_data
            logging.debug(f"{os.getpid()} got next batch of size {len(next_batch_data)}")
            
            # Gather images of the same shape, as images of variying shapes cannot be inside the same batch
            while len(batch_data) > batch_size:
                next_batch, batch_data = filter_batch(batch_data, max_size=batch_size)
                images, paths = zip(*next_batch)
                logging.debug(f"{os.getpid()} putting next batch into queue")
                batch_queue.put(QueueEntry(image_data=images, output_paths=paths))
                logging.debug(f"{os.getpid()} done")
                
            if is_finished:
                while len(batch_data) > 0:
                    next_batch, batch_data = filter_batch(batch_data, max_size=batch_size)
                    images, paths = zip(*next_batch)
                    batch_queue.put(QueueEntry(image_data=images, output_paths=paths))
                logging.info(f"{os.getpid()} dataloading finished")
                batch_queue.put(QueueEntry(is_stop_entry=True))
                return
            
            if not loop_until_complete:
                return
            else:
                # Wait until previous batches have been consumed before loading further images
                while batch_queue.qsize() > 3:
                    logging.debug(f"{os.getpid()} waiting for batch queue to reduce in size: {batch_queue.qsize()}")
                    time.sleep(1)
    except:
        batch_queue.put(QueueEntry(is_stop_entry=True, failed=True))
        logging.exception("Error occurred during dataloading.")

def datawriter():
    while True:
        try:
            job = output_queue.get()
            if job.is_stop_entry:
                completed_queue.put(OutputJob(is_stop_entry=True))
                break
            job.execute()
            if job.output_path:
                completed_queue.put(OutputJob(output_path=job.output_path))
            
                
        except Exception as e:
            logging.exception(f"Exception occurred in data writer: {str(e)}")
            
    

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
    shuffle: bool = False
    """Whether to process images in random order."""
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

    def get_next_batch(self, timeout_in_s=30) -> QueueEntry:
        if not self.parallel_dataloading:
            # Manually generate the next element in the queue
            dataloader(batch_size=self.batch_size, input_output_path_iterator=self.data_iter,
                       ignore_existing=self.overwrite_existing)
        return batch_queue.get()
            
    def prepare_dataloading(self, img_paths: List[Path], input_path, output_path):
        input_output_paths = [(img_path, output_path / img_path.relative_to(input_path))
                              for img_path in (reversed(img_paths) if self.reversed_processing else img_paths)]
        
        if self.compression_quality != -1:
            # Set output file type to jpg if compression is active
            input_output_paths = [(img, path.with_suffix(".jpg")) for img, path in input_output_paths]
            
        if self.shuffle:
            random.shuffle(input_output_paths)  # random.shuffle is an inplace operation
            
        self.data_iter = iter(input_output_paths)
        
        if self.parallel_dataloading:
            logging.info("Starting data worker")
            self.dataloader = multiprocessing.Process(
                target=dataloader, 
                kwargs=dict(batch_size=self.batch_size, input_output_path_iterator=self.data_iter,
                            ignore_existing=self.overwrite_existing)
            )
            self.dataloader.daemon = True
            atexit.register(lambda: self.dataloader.terminate())
            self.dataloader.start()
            
        self.datawriter = multiprocessing.Process(target=datawriter, daemon=True)
        self.datawriter.daemon = True
        self.datawriter.start()
        atexit.register(lambda: self.datawriter.terminate())
              
    def prepare_anonymization(self, input_path, file_types, output_path, start=0, stop=None, step=1):
        logging.info(f'Anonymizing images in {input_path} and saving the anonymized images to {output_path}...')

        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        assert output_path.is_dir(), 'Output path must be a directory'

        logging.info("Gathering input image paths... ")

        if self._img_paths_ is None:
            self._img_paths_ = []
            for file_type in file_types:
                matches = sorted(Path(input_path).glob(f'**/*.{file_type}'))
                self._img_paths_.extend(matches[slice(start, stop, step)])
        else:
            self._img_paths_ = self._img_paths_[slice(start, stop, step)]

        logging.info("Done. Found {len(self._img_paths_)} to process.")
        self.prepare_dataloading(self._img_paths_, input_path, output_path)        


    def process_batch(self, batch, detection_thresholds):
        if batch.image_data is not None and batch.image_data[0] is not None:  # image data is set to None for batches that don't require processing
            # Anonymize image
            anonymized_images_detections = self.anonymize_images_np(batch.image_data, detection_thresholds)
            
            for idx, (anonymized_image, detections) in enumerate(anonymized_images_detections):
                
                job = OutputJob(image_data=anonymized_image, output_path=batch.output_paths[idx],
                                compression_quality=self.compression_quality)
                if self.save_detection_json_files:
                    job.detections = detections
                    
                logging.debug(f"{os.getpid()} putting job into output queue")
                output_queue.put(job)
                logging.debug(f"{os.getpid()} put job into queue")
        else:
            if batch.output_paths is not None:
                for output_path in [p for p in batch.output_paths if p is not None]:
                    output_queue.put(OutputJob(output_path=output_path))

    def anonymize_images(self, input_path, output_path, detection_thresholds, file_types, start=0, stop=None, step=1):
        self.prepare_anonymization(input_path, file_types, output_path, start, stop, step)
        if len(self._img_paths_) == 0:
            logging.info("No images to process")
            output_queue.put(OutputJob(is_stop_entry=True))
            return 
        # use progiter instead of tqdm since it plays nicer with multiprocessing (tqdm apparently isn't threadsafe)
        progress = ProgIter(self._img_paths_, desc="/".join(self._img_paths_[0].parts[-4:-1]), stream=sys.stderr)
        last_permil = -1
        progress_iter = iter(progress)
        try:
            while True:
                logging.debug(f"{os.getpid()} getting next batch")
                next_batch = self.get_next_batch()
                logging.debug(f"{os.getpid()} done. processing next batch")
                self.process_batch(next_batch, detection_thresholds)
                logging.debug(f"{os.getpid()} batch finished processing.")    
                # Iterate over the progress iterator to log the progress
                for _ in range(len(next_batch.output_paths or [])):
                    next(progress_iter, None)
                logging.debug("")
                
                permil = int((progress._now_idx + 1) / progress.total * 1000)
                if permil != last_permil:
                    last_permil = permil
                    logging.info(progress.format_message())
                
                if next_batch.failed:
                    raise RuntimeError("Dataloader failed")
                
                if next_batch.is_stop_entry:
                    for _ in progress_iter:  # finalize the progress log
                        pass
                    logging.info(progress.format_message())
                    logging.info("\nfinished" + "/".join(self._img_paths_[0].parts[-4:-1]))
                    break
            output_queue.put(OutputJob(is_stop_entry=True))
                    
        finally:
            while output_queue.qsize() > 0:
                logging.info(f"Waiting for output queue to finish: {output_queue.qsize()}")
                time.sleep(.5)
            logging.info("Output queue finished.")
            
            completed: List[OutputJob] = []
            
            logging.info(f"Completed queue size: {completed_queue.qsize()}")
            while completed_queue.qsize() > 0:
                job = completed_queue.get(block=False)
                if not job.is_stop_entry:
                    completed.append(job)
                    
            logging.debug(f"Completed Jobs: {len(completed)}")
                    
            if completed:
                completed_per_folder = {job.output_path.parent: [] for job in completed}
                for job in completed:
                    completed_per_folder[job.output_path.parent].append(job)
                
                logging.debug(f"Completed folders: {list(completed_per_folder)}")    
                    
                for folder, jobs in completed_per_folder.items():
                    completed_file = folder / "completed.txt"
                    previously_completed = set(completed_file.read_text().split("\n") if completed_file.exists() else [])
                    if not all(job.output_path.stem in previously_completed for job in jobs):
                        logging.info(f"Writing {completed_file}")
                        completed_image_stems = {
                            *(job.output_path.stem for job in jobs), 
                            *(stem for stem in previously_completed if len(stem) > 0)
                        }
                        completed_file.write_text('\n'.join(sorted(completed_image_stems)))
                    
        logging.info("Anonymization complete.")