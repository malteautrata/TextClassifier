# TextClassifier
Different models for text classification of German newspaper articles

Dataset: https://tblock.github.io/10kGNAD/

### Bert Metrics:
* lr = 2e-7, gradient_clip = True
* 10% dropout
* batch size: 8
* loss logged every 50 steps
* eval accuracy logged every 200
* duration 448.15 minutes
* ![Training metrics](results/bert_results/metrics/graph_20_epochs.png)
* Testaccuracy: 90.27%, took 0.7174 minutes
* Testaccuracy distribution:
* ![Test distribtuon](results/bert_results/metrics/test_evaluation.png)
* Confusion matrix:
* ![Confusion matrix](results/bert_results/metrics/confusion_matrix.png)

### T5 Encoder Metrics:
* lr = 2e-6
* batch_size = 8
* loss logged every 50 steps
* eval accuracy logged every 200
* duration: 258.08 minutes
* ![Training metrics](results/t5_results/encoder/metrics/graph_10_epochs.png)
* seems to overfit
* Testaccuracy: 90.27%, took 1.00 minute
* Testaccuracy distribution:
* ![Test distribtuon](results/t5_results/encoder/metrics/test_evaluation.png)
* Confusion matrix:
* ![Confusion matrix](results/t5_results/encoder/metrics/confusion_matrix.png)


### T5 Metrics:
* batch size = 8
* epoch 0-9: lr = 2e-5
* epoch 9-14: lr = 2e-6
* loss logged every 50 steps
* eval accuracy logged ever 200 steps
* duration: 152.26 minutes
* ![Training metrics](results/t5_results/transformer/metrics/graph_15_epochs.png)
* Testaccuracy: 88.72%, took 0.38 minutes
* Testaccuracy distribution:
* ![Test distribtuon](results/t5_results/transformer/metrics/test_evaluation.png)
* Confusion matrix:
* ![Confusion matrix](results/t5_results/transformer/metrics/confusion_matrix.png)

### Llama 3 instruct:
* batch size = 4
* lr = 2.0e-5
* loss logged every 100 steps
* 1 epoch
* duration: 370 minutes (6.17 hours)
* ![Training metrics](results/llama3_results/instruct/metrics/graph_1_epoch.png)
* Testaccuracy: 87.94%, took 27 minutes
* Testaccuracy distribution:
* ![Test distribtuon](results/llama3_results/instruct/metrics/test_evaluation.png)
* Confusion matrix:
* ![Confusion matrix](results/llama3_results/instruct/metrics/confusion_matrix.png)

### Llama 3 Classification head:
* batch size = 4
* steps 0-1200: no weight decay, lr = 1e-4
* steps 1200-3000: weight decay = 0.01, lr = 1e-4
* steps 3000-4300: weight decay = 0.01, lr = 1e-5
* duration: 53.05 hours hours
* ![Training metrics](results/llama3_results/classification_head/metrics/training_loss.png)
* ![Training metrics](results/llama3_results/classification_head/metrics/training_accuracy.png)
* Testset evaluation duration: 31 minutes
* Testaccuracy: 92.32%
* F1-Score: 92.28%
* Testaccuracy distribution:
* ![Training metrics](results/llama3_results/classification_head/metrics/test_evaluation.png)
* Confusion matrix:
* ![Training metrics](results/llama3_results/classification_head/metrics/confusion_matrix.png)
