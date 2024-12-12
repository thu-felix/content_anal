import csv
import datasets
import os
import random

from openprompt.data_utils import InputExample, InputFeatures, FewShotSampler
from openprompt.data_utils.data_processor import DataProcessor
from openprompt.utils.logging import logger
from yacs.config import CfgNode

class FakeRealDataProcessor(DataProcessor):
    def __init__(self, labels=['fake, real'], labels_path=None):
        super().__init__(labels, labels_path)
        random.seed(42)
        self.examples = []

    def get_examples(self, data_dir=None, split=None):
        if split == "valid" or split == "dev":
            split = "validation"
        
        if data_dir == None:
            return []

        if self.examples == []:
            data_info = {
                "fake" : "Fake.csv",
                "real" : "Real.csv"
            }

            for label in data_info.keys():
                with open(os.path.join(data_dir, data_info[label])) as f:
                    reader = csv.reader(f)
                    col_names = next(reader)

                    for row in reader:
                        row_data = dict(zip(col_names, row))
                        text_a = row_data['text']
                        del row_data['text']

                        self.examples.append(InputExample(
                            text_a = text_a,
                            label = label,
                            meta = row_data
                        ))

            random.shuffle(self.examples)
        return random.sample(self.examples, int(len(self.examples) / 3))

class IMDBDataProcessor(DataProcessor):
    def __init__(self, labels=['positive, negative'], labels_path=None):
        super().__init__(labels, labels_path)

    def get_examples(self, data_dir='stanfordnlp/imdb', split='train'):
        if split == "valid" or split == "dev":
            split = "validation"

        if data_dir == None:
            return []

        ds = datasets.load_dataset(data_dir, split=split)

        return list(map(lambda data: InputExample(text_a = data['text'], label = 'negative' if data['label'] == 0 else 'positive'), ds))

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
        topics = ', '.join([topic[0] for topic in list(zip(self.topic_labels, data['gold_label_list'])) if topic[1] == 1])

        text_a = data['text']

        return InputExample(
            text_a = text_a,
            tgt_text = topics,
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

