## MFF-FTNet Result
<img width="498" alt="Snipaste_2024-11-26_09-52-21" src="https://github.com/user-attachments/assets/2a7b37be-7779-4fe3-a673-3e4ffe3672ac">
<img width="498" alt="Snipaste_2024-11-26_09-52-33" src="https://github.com/user-attachments/assets/b9050a7c-d0d4-4dcd-8ebe-be5bd9a8fdc0">

## Requirements
1. Install Python 3.8, and the required dependencies.
2. Required dependencies can be installed by: ```pip install -r requirements.txt```
## Data
The datasets can be obtained and put into `datasets/` folder in the following way:
* [3 ETT datasets](https://github.com/zhouhaoyi/ETDataset) should be placed at `datasets/ETTh1.csv`, `datasets/ETTh2.csv` and `datasets/ETTm1.csv`.
* [ETTm2 dataset](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) placed at `datasets/ETTm2.csv`
* [Weather dataset](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR) placed at `datasets/WTH.csv`
## Usage
To train and evaluate MFF-FTNet on a dataset, run the script from the scripts folder: ```./scripts/model_ETT.sh``` or ```./scripts/model_WTH.sh``` 
## Acknowledgements
The implementation of MFF-FTNet builds upon resources from the following codebases and repositories. We express our gratitude to the original authors for generously making their work available as open-source.
* https://github.com/salesforce/CoST
* https://github.com/xingyu617/SimTS_Representation_Learning

