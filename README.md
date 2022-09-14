# BIGCONTEST CODE


## Get Started

1. Install Python 3.8. (i.g, Create environment `conda create -n bigcon python=3.8`)
2. Download data. `See Dropbox`
3. Download requirement packages ***pip install -r requirements.txt*** 
4. See preprocessing file `1_preprocessing.ipynb`


## Label encoding 

1. income_type : `['EARNEDINCOME' 'EARNEDINCOME2' 'FREELANCER' 'OTHERINCOME' 'PRACTITIONER' 'PRIVATEBUSINESS']`

2. employment_type : `['계약직' '기타' '일용직' '정규직']`

3. houseown_type : `['기타가족소유' '배우자' '자가' '전월세']`

4. purpose : `['기타' '대환대출' '사업자금' '생활비' '자동차구입' '전월세보증금' '주택구입' '투자']`

## Base train test split

`from sklearn.model_selection import train_test_split`
`train, val = train_test_split(full_data, test_size=0.20,   random_state=20205289,  stratify=full_data['is_applied'])`
`train, test = train_test_split(train, test_size=0.125,   random_state=20205289,  stratify=train['is_applied'])`
`train.to_csv("./prepro_data/train.csv", index = False)`
`val.to_csv("./prepro_data/val.csv", index = False)`
`test.to_csv("./prepro_data/test.csv", index = False)`
