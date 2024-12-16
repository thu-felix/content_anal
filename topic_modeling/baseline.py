pip install transformers datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
import torch
from nltk.translate.bleu_score import SmoothingFunction

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-topic-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-topic-latest")

# Define the data processor class
class TweetTopicDataProcessor:
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

    def convert_data(self, data):
        topics = ', '.join([topic[0] for topic in list(zip(self.topic_labels, data['gold_label_list'])) if topic[1] == 1])
        text_a = data['text']
        return {"text": text_a, "topics": topics}

    def get_examples(self, data_dir='cardiffnlp/super_tweeteval', split='train'):
        if split == "valid" or split == "dev":
            split = "validation"
        
        if data_dir is None:
            return []

        ds = load_dataset(data_dir, 'tweet_topic', split=split)
        return [self.convert_data(data) for data in ds]

# Load and preprocess dataset
processor = TweetTopicDataProcessor()
dataset = processor.get_examples(split="valid")

# Helper function to predict topics
def predict_topics(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()
    predictions = torch.sigmoid(torch.tensor(logits)).numpy() > 0.5
    predicted_labels = [processor.topic_labels[i] for i in range(len(predictions[0])) if predictions[0][i]]
    return predicted_labels

# Evaluate the model using sentence_bleu
smoothing = SmoothingFunction().method1
scores = []
for example in dataset:
    predicted_topics = predict_topics(example["text"])
    reference_topics = example["topics"].split(', ')
    score = sentence_bleu([reference_topics], predicted_topics, weights=(1.0, 0, 0, 0), smoothing_function=smoothing)  # unigram BLEU with smoothing
    scores.append(score)

# Calculate and print average BLEU score
average_bleu = sum(scores) / len(scores)
print(f"Average BLEU score: {average_bleu:.4f}")