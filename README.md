Content Analysis
================

Set of configuration files, dataset loaders, and runner scripts for training OpenPrompt model for content analysis tasks.[
Training configuration and parameters are stored in a `yaml` file in accordance to [OpenPrompt Configuration GUide](https://thunlp.github.io/OpenPrompt/notes/configuration.html) located in each task's respective directories, which are then loaded by `cli.py` to do the training and validation process.

These tasks include
### Sentiment Analysis
```bash
cd sentim
../cli.py --config_yaml classification_PN.yaml
```

### Fake News Detection
```bash
cd fakenews
../cli.py --config_yaml config_t5.yaml
```

### Topic Modeling
```bash
cd topic_modeling
../cli.py --config_yaml config_t5_full.yaml
```
In the each task's directories also stored python notebooks on running the openprompt powered model training, they also included python notebooks or scripts to run a baseline model to compare the results with the PromptModel.

###Model Test
```bash
1.Download Model_Test.ipynb file in main branch and open in Google Colab.
2.Download the checkpoint files of trained models. (Download Link: [https://drive.google.com/drive/folders/1es6UDG_TAD3-1DiL_Aj1H2iyNhwBVqoY?usp=sharing](https://drive.google.com/file/d/1LkSeshTw4pzuJUYyAtoqUe_o4QuLf7sO/view?usp=sharing) )
3.Run the code snippets from top to bottom.

