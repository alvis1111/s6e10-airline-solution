# 数据说明

数据不提交到 Git。需自行下载。

## 比赛数据

`kaggle competitions download -c playground-series-s6e10`

放 `data/playground-series-s6e10/` 下：train.csv / test.csv / sample_submission.csv

## 原始数据（航线画像的 `orig_proba` 特征需要）

Kaggle Dataset `teejmahal20/airline-passenger-satisfaction`（129k 真实乘客，合成数据的来源）。

```python
import kagglehub
path = kagglehub.dataset_download('teejmahal20/airline-passenger-satisfaction')
```

脚本 `route_profile.py` 里 `REAL` 变量指向该路径，改成你的下载位置即可。
