Content Analysis
================

Set of configuration files, dataset loaders, and runner scripts for training OpenPrompt model for content analysis tasks, such as

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
