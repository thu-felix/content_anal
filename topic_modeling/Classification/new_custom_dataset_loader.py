import csv
import datasets
import os
import random

from openprompt.data_utils import InputExample, InputFeatures, FewShotSampler
from openprompt.data_utils.data_processor import DataProcessor
from openprompt.utils.logging import logger
from yacs.config import CfgNode

class FakeRealDataProcessor(DataProcessor):
    def __init__(self, labels=['fake', 'real'], labels_path=None):
        super().__init__(labels, labels_path)
        random.seed(42)
        self.examples = []

    def get_examples(self, data_dir=None, split=None):
        if split in ["valid", "dev"]:
            split = "validation"
        if data_dir is None:
            return []

        if not self.examples:
            data_info = {0: "Fake.csv", 1: "True.csv"}
            for label, filename in data_info.items():
                file_path = os.path.join(data_dir, filename)
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        text_a = row.get('text', None)
                        if not text_a:
                            logger.warning("Missing 'text' column in row, skipping.")
                            continue
                        self.examples.append(
                            InputExample(text_a=text_a, label=label, meta=row)
                        )
            random.shuffle(self.examples)

        return self.examples


class IMDBDataProcessor(DataProcessor):
    def __init__(self, labels=['positive', 'negative'], labels_path=None):
        super().__init__(labels, labels_path)

    def get_examples(self, data_dir='stanfordnlp/imdb', split='train'):
        if split == "valid" or split == "dev":
            split = "test"
        #if s
        if data_dir == None:
            return []

        ds = datasets.load_dataset(data_dir, split=split)

        return list(map(lambda data: InputExample(text_a = data['text'], label = data['label']), ds))

class TweetTopicDataProcessor(DataProcessor):
    def __init__(self, labels=None, labels_path=None):
        if not labels:
            self.topic_labels = (
                'arts_&_culture',
                'business_&_entrepreneurs',
                'celebrity_&_pop_culture',
                'diaries_&_daily_life',
                'family',
                'fashion_&_style',
                'film_tv_&_video',
                'fitness_&_health',
                'food_&_dining',
                'gaming',
                'learning_&_educational',
                'music',
                'news_&_social_concern',
                'other_hobbies',
                'relationships',
                'science_&_technology',
                'sports',
                'travel_&_adventure',
                'youth_&_student_life',
            )
            labels = self.topic_labels
        else:
            self.topic_labels = labels

        super().__init__(labels, labels_path)

    def convert_data(self, data):
        topics = [topic for topic, is_present in zip(self.topic_labels, data['gold_label_list']) if is_present == 1]
        label = topics[0] if topics else "other_hobbies"  # Default to a category if none are present
    
        text_a = data['text']
    
        return InputExample(
            text_a=text_a,
            label=self.labels.index(label)  # Convert to index for classification
        )
        
    def get_examples(self, data_dir='cardiffnlp/super_tweeteval', split='train'):
        if split == "valid" or split == "dev":
            split = "validation"
        
        if data_dir == None:
            return []

        ds = datasets.load_dataset(data_dir,'tweet_topic', split=split)
        return list(map(self.convert_data, ds))

PROCESSORS = {
    "fakenews": FakeRealDataProcessor,
    "imdb": IMDBDataProcessor,
    "tweet_topic" : TweetTopicDataProcessor,
}

def load_dataset(config: CfgNode, return_class=True, test=False):
    r"""A plm loader using a global config.
    It will load the train, valid, and test set (if exists) simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.
        return_class (:obj:`bool`): Whether return the data processor class
                    for future usage.

    Returns:
        :obj:`Optional[List[InputExample]]`: The train dataset.
        :obj:`Optional[List[InputExample]]`: The valid dataset.
        :obj:`Optional[List[InputExample]]`: The test dataset.
        :obj:"
    """
    dataset_config = config.dataset

    processor = PROCESSORS[dataset_config.name.lower()]()

    train_dataset = None
    valid_dataset = None
    if not test:
        try:
            train_dataset = processor.get_train_examples(dataset_config.path)
        except FileNotFoundError:
            logger.warning(f"Has no training dataset in {dataset_config.path}.")
        try:
            valid_dataset = processor.get_dev_examples(dataset_config.path)
        except FileNotFoundError:
            logger.warning(f"Has no validation dataset in {dataset_config.path}.")

    test_dataset = None
    try:
        test_dataset = processor.get_test_examples(dataset_config.path)
    except FileNotFoundError:
        logger.warning(f"Has no test dataset in {dataset_config.path}.")
    # checking whether downloaded.
    if (train_dataset is None) and \
       (valid_dataset is None) and \
       (test_dataset is None):
        logger.error("Dataset is empty. Either there is no download or the path is wrong. "+ \
        "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()
    if return_class:
        return train_dataset, valid_dataset, test_dataset, processor
    else:
        return  train_dataset, valid_dataset, test_dataset

