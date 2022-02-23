# multi-model-classification

## classification model 3개 사용

- mask 착용여부 한번
- 성별 한번
- Age 한번

=> 3개의 결과값을 합쳐서 최종 클래스 결정

## train

- 3개의 classification을 위해 3개의 dataset가 필요하고, dataset.py에 정의되어 있음

```python
# mask classification
python train-mask.py
```

```python
# gender classification
python train-gender.py
```

```python
# age classification
python train-age.py
```

## inference

```python
python inference.py
```

* test set에서의 오답 확인

```python
python inference-train.py
```
