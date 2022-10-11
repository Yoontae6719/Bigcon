# 앱 사용성 데이터를 통한 대출신청 예측 분석


## Overview

![image](https://user-images.githubusercontent.com/87846187/195023245-b59ced62-367c-4067-8090-be344bab5357.png)


## Results

- The results for the forecast after June can be found at the following Google Drive link: [Google Drive](https://drive.google.com/drive/folders/17p9EDoJe_3AdiCIrTWRsN57S32CtrocY?usp=sharing)
- [x] test.csv
- [x] cluster_user.csv 

## Notice

You ***MUST*** Create the following folders in advance.
- [x] data
- [x] prepro_data
- [x] DL_dataset

The `data folder` requires the ***loan_result.csv***, ***log_data.csv***, and ***user_spec.csv*** files. [Google Drive](https://drive.google.com/file/d/1UNViyS4iHeTkAAR9yO9iwB5zrQKuf4w3/view?usp=sharing)  \
When you run `1_preprocessing_real.ipynb`, the preprocessed file is stored in the `prepro_data folder` in the following two forms:
- [x] full_data.csv
- [x] submit_test.csv 

full_data.csv means a dataset for learning before June and a dataset for submit_test.csv means a dataset for testing after June.



Next, when `2_Preprocessing_2.ipynb` and `3_Preprocessing_3.ipynb` are runed, the data on the user's behavior is reflected in ***loon_result.csv***. At this time, ray was used for parallel processing. The reflected results are similarly stored in the `prepro_data folder` as `full_data.csv` and `submit_test.csv`.

Thired, by runing `4_Preprocessing_4.ipynb`, continuous variables are converted into categorical variables and stored in a `dataset folder`. The stored data sets are as follows.
- [x] full_data.csv
- [x] submit_test.csv 

Finally, a dataset for deep learning is configured by executing `6_DL_models_inputs.ipynb`. The following data is stored in the `DL_dataset`.
- [x] fold_0.csv
- [x] fold_1.csv
- [x] fold_2.csv
- [x] fold_3.csv
- [x] fold_4.csv
- [x] train.csv
- [x] test.csv

Through deep learning, all results are stored in `DL_dataset`.

## Get Started

1. Install Python 3.8. (i.g, Create environment ***conda create -n bigcon python=3.8***)
2. Download data. (first, you must be load data in ./data folder)
3-1. Download requirement packages ***pip install -r requirements.txt*** 
3-2. Download Autogolun packages ***pip3 install autogluon*** you can use GPU mode See the [link](https://auto.gluon.ai/stable/install.html).
3-3. Download Ray package for preprocessing ***pip install ray*** 
4. For the preprocessing process, run five jupyer notes as follows.
- [x] 1_preprocessing_real.ipynb
- [x] 2_Preprocessing_2.ipynb 
- [x] 3_Preprocessing_3.ipynb
- [x] 4_Preprocessing_4.ipynb
- [x] 6_DL_models_inputs.ipybn
5. To run the ML model, run the following jupyter notebook. The weights of all models can be downloaded from the following Google drive Link : [Google Drive](https://drive.google.com/file/d/1-sMeVVD-MjW48fmO6xdK0SVV-6sGutOn/view?usp=sharing)

- [x] 5_test_modeling-ACC-ALL.ipynb
6. Train the model. We provide the experiment scripts of all benchmarks under the folder `./runfile`. The weights of Deep learning can be downloaded from the `checkpoints.zip folder`. You can reproduce the experiment results by:
```bash
bash ./runfile/big_1.sh
```

7. For machine learning and deep learning models, run Voting enamble 7_ML_DL_model_output.ipybn. All results are stored in the `submit folder`.

8. Run 8_Clustering.ipynb for clustering results. All results are stored in the `submit folder`

## Results

- The results for the forecast after June can be found at the following Google Drive link: [Google Drive](https://drive.google.com/drive/folders/17p9EDoJe_3AdiCIrTWRsN57S32CtrocY?usp=sharing)
- [x] test.csv
- [x] cluster_user.csv 


### Models

- We construct a final model with an ensemble of machine learning models and deep learning models. 


![image](https://user-images.githubusercontent.com/87846187/195022823-44a2d855-b8d8-4b66-af76-5d84f548a5d1.png)

![image](https://user-images.githubusercontent.com/87846187/195022926-cc0fc4b7-e881-40eb-b019-09b8c736e665.png)


### Clustering

- Clustering was conducted from the Embedding vector extracted from deep learning, and an evaluation index based on cumulative probability distribution was created to find the optimal cluster from the Embedding vector of high dimensions.


![image](https://user-images.githubusercontent.com/87846187/195023653-3deede11-55e0-41bd-8b55-436629178e71.png)


## Contact

If you have any questions or want to use the code, please contact `yoontae@unist.ac.kr`.
