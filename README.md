##  [ConcreteDataset](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)
- data: `origin/Concrete_Data.xls`
- analytics: `report.pdf`
###  Environment
- Windows 10
- Python 3.7
- Anaconda (recommended)
###  Install Package
You should install all package
scikit-learn, numpy, pandas, xlrd, matplotlib, seaborn, joblib, PyYAML
```shell script
> conda install scikit-learn numpy pandas xlrd matplotlib seaborn joblib
> pip install PyYAML
```
You can use pip but anaconda is recommended on windows 10
###  Execute
####  baseline
- Linear Model - ElasticNet & Not feature extraction
```shell script
> python main.py baseline
```
#### processed
- Linear Model - ElasticNet & feature extraction
```shell script
> python main.py processed
```
#### advanced
- Ensemble Model - GradientBoosting & feature extraction
```shell script
> python main.py advanced
```
