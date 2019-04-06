#!/usr/bin/python
import argparse
import glob, os
from shutil import copyfile
import re
from sets import Set

class REMatcher(object):
    """ regular expression matcher
    """
    def __init__(self, regexp):
        self.regexp = re.compile(regexp, re.IGNORECASE)

    def match(self,match_string):
        self.results = self.regexp.match(match_string)
        return bool(self.results)

    def group(self,i):
        return self.results.group(i)

if __name__ == "__main__":
    # init command-line argument parser:
    parser = argparse.ArgumentParser(description='Convert Tibetan MNIST into DIGITS dataset format.')
    
    parser.add_argument(
        '--directory', 
        type=str, 
        help='root directory of Tibetan MNIST dataset.'
    )
    parser.add_argument(
        '--format', 
        type=str, 
        help='Tibetan MNIST dataset image record format.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        help='DIGITS dataset output path.'
    )

    args = parser.parse_args()    

    # parse arguments:
    tibetan_mnist_root_dir = args.directory
    image_format = args.format
    digits_dataset_output = args.output

    # create digits dataset:
    if not os.path.exists(digits_dataset_output):
        os.makedirs(digits_dataset_output)

    # create label parser:
    label_parser = REMatcher(
        "(\d+)_(\d+)_(\d+).{}".format(image_format)
    )

    # label set:
    label_set = Set([])

    # train file list:
    train_file_list = []

    for i, record in enumerate(
        glob.glob(
            os.path.join(tibetan_mnist_root_dir, "*.{}".format(image_format))
        )
    ):
        # parse label:
        record = os.path.split(record)[-1]

        if label_parser.match(record):
            label, page_id, line_id = label_parser.group(1), label_parser.group(2), label_parser.group(3)

            if not label in label_set:
                label_set.add(label)
                os.makedirs(os.path.join(digits_dataset_output, label))
            
            copyfile(
                os.path.join(tibetan_mnist_root_dir, record), 
                os.path.join(digits_dataset_output, label, "{:05d}.{}".format(i, image_format))
            )

            train_file_list.append(
                "{} {}".format(
                    os.path.join("/data/images/classification/tibetan-mnist", label, "{:05d}.{}".format(i, image_format)),
                    label
                )
            )
        else:
            print record
            
    # labels.txt
    with open(
        os.path.join(digits_dataset_output, "labels.txt"), 
        'w'
    ) as f:
        f.write("\n".join(label_set))

    # train.txt
    with open(
        os.path.join(digits_dataset_output, "train.txt"), 
        'w'
    ) as f:
        f.write("\n".join(train_file_list))