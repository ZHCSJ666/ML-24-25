```python
import rootutils

rootutils.setup_root(".", indicator=".project-root", pythonpath=True)

from src.metrics.bleu import SacreBLEUScore

bleu = SacreBLEUScore(n_gram=1, tokenize="none")

preds = [
    "hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world",
    "hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world",
]
target = [
    "hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world",
    "hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world",
]

# for i in range(len(preds)):
#     bleu.update(preds[i], target[i])
```


```python
bleu(preds[1], target[1])
```


```python
from src.metrics.rouge import ROUGEScore

rouge = ROUGEScore(rouge_keys="rouge1")["rouge1_fmeasure"]
rouge(preds[1], target[1])
```
