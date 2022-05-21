"""
Copyright 2018 understand.ai

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import argparse
import socket
import os
import sys
from pathlib import Path


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # error messages only


def setup_logging(logfile, level=logging.INFO):
    print(f"Logging to {logfile}")
    if not logfile.parent.exists():
        logfile.parent.mkdir(parents=True)
        os.system(f"chmod -R 0777 {logfile.parent}")
        
    pid = os.getpid()
    
    logging.basicConfig(filename=logfile, filemode='a',
                        format=f'%(asctime)s %(levelname)s {pid}: %(message)s',
                        datefmt='%H:%M:%S', level=level)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    

def parse_args():
    parser = argparse.ArgumentParser(
        description='Anonymize faces and license plates in a series of images.')
    parser.add_argument('--input', required=True,
                        metavar='/path/to/input_folder',
                        help='Path to a folder that contains the images that should be anonymized. '
                             'Images can be arbitrarily nested in subfolders and will still be found.')
    parser.add_argument('--image-output', required=True,
                        metavar='/path/to/output_foler',
                        help='Path to the folder the anonymized images should be written to. '
                             'Will mirror the folder structure of the input folder.')
    
    parser.add_argument('--start', help="Index to position in the list of images where to start processing.", 
                        default=0, type=int)
    parser.add_argument('--stop', help="Index to position in the list of images where to stop processing (exclusive index)", 
                        default=None, type=int)
    parser.add_argument('--step', help="Step size to apply to image processing to only process every nth image.", 
                        default=1, type=int)
    parser.add_argument('--weights', required=True,
                        metavar='/path/to/weights_foler',
                        help='Path to the folder where the weights are stored. If no weights with the '
                             'appropriate names are found they will be downloaded automatically.')
    parser.add_argument('--image-extensions', required=False, default='jpg,png',
                        metavar='"jpg,png"',
                        help='Comma-separated list of file types that will be anonymized')
    parser.add_argument('--face-threshold', type=float, required=False, default=0.3,
                        metavar='0.3',
                        help='Detection confidence needed to anonymize a detected face. '
                             'Must be in [0.001, 1.0]')
    parser.add_argument('--plate-threshold', type=float, required=False, default=0.3,
                        metavar='0.3',
                        help='Detection confidence needed to anonymize a detected license plate. '
                             'Must be in [0.001, 1.0]')
    parser.add_argument('--write-detections', dest='write_detections', action='store_true')
    parser.add_argument('--no-write-detections', dest='write_detections', action='store_false')
    parser.set_defaults(write_detections=True)
    parser.add_argument('--obfuscation-kernel', required=False, default='21,2,9',
                        metavar='kernel_size,sigma,box_kernel_size',
                        help='This parameter is used to change the way the blurring is done. '
                             'For blurring a gaussian kernel is used. The default size of the kernel is 21 pixels '
                             'and the default value for the standard deviation of the distribution is 2. '
                             'Higher values of the first parameter lead to slower transitions while blurring and '
                             'larger values of the second parameter lead to sharper edges and less blurring. '
                             'To make the transition from blurred areas to the non-blurred image smoother another '
                             'kernel is used which has a default size of 9. Larger values lead to a smoother '
                             'transition. Both kernel sizes must be odd numbers.')
    parser.add_argument('--compression-quality', default=-1, required=False,
                        help='level of jpeg compression for storing resulting images. If not provided, '
                        'images will be stored with the same filetype as the originating files and '
                        'with the default compression level (if applicable).')
    parser.add_argument('--reversed-processing', action='store_true',
                        help='Process images in reverse order, useful for parallel runs on two machines')
    parser.add_argument('--shuffle', help="Process images in random order.", action='store_true')
    parser.add_argument('--batch-size', '-b', help="maximum number of images to process inside a batch",
                        required=False, type=int, default=8)
    parser.add_argument('--gpu-memory-limit', help="Limit (in percent) of memory to be used per GPU, e.g. 50", 
                        type=int, default=100)
    parser.add_argument('--logfile', help="File where to log messages (opened in append mode)",
                        default=f"/tmp/anonymizer_logs/{socket.gethostname()}.{os.getpid()}.log")
    parser.add_argument('--debug', help="Turn on debug logging", required=False, action='store_true')

    args = parser.parse_args()

    setup_logging(Path(args.logfile), level=logging.DEBUG if args.debug else logging.INFO)

    return args


def main():
    args = parse_args()

    try:
        if args.debug:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        # above environment configuration needs to be set before imports, therefore not defined at module level
        from anonymizer.anonymization import Anonymizer
        from anonymizer.detection import Detector, download_weights, get_weights_path
        from anonymizer.obfuscation import Obfuscator

        download_weights(download_directory=args.weights)

        kernel_size, sigma, box_kernel_size = args.obfuscation_kernel.split(',')
        obfuscator = Obfuscator(kernel_size=int(kernel_size), sigma=float(sigma), box_kernel_size=int(box_kernel_size))

        detectors = {
            'face': Detector(kind='face', weights_path=get_weights_path(args.weights, kind='face'),
                            gpu_memory_fraction=args.gpu_memory_limit / 220 if args.gpu_memory_limit != 100 else None),
            'plate': Detector(kind='plate', weights_path=get_weights_path(args.weights, kind='plate'),
                            gpu_memory_fraction=args.gpu_memory_limit / 220 if args.gpu_memory_limit != 100 else None)
        }
        detection_thresholds = {
            'face': args.face_threshold,
            'plate': args.plate_threshold
        }
        anonymizer = Anonymizer(obfuscator=obfuscator, detectors=detectors, batch_size=args.batch_size,
                                compression_quality=args.compression_quality, save_detection_json_files=args.write_detections,
                                reversed_processing=args.reversed_processing, parallel_dataloading=True, shuffle=args.shuffle)
        anonymizer.anonymize_images(input_path=args.input, output_path=args.image_output, start=args.start, stop=args.stop, step=args.step,
                                    detection_thresholds=detection_thresholds, file_types=args.image_extensions.split(','))
    except Exception as e:
        logging.exception("Exception occurred during anonymization")


if __name__ == '__main__':
    main()

