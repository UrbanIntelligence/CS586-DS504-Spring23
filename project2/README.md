# Individual Project 2
# Classification with Deep Learning
#### Due Date
* Updated to Tuesday Mar 14, 2023 (23:59), 


#### Total Points
* 100 (One Hundred)

## Goal
In this project, you will be asked to finish a sequence classification task using deep learning. A trajectory data set with five taxi drivers' daily driving trajectories in 6 months will be provided, and the task is to build a model to predict which driver a trajectory belongs to. Trajectory to be classified include all GPS records of a driver in one day. During the test, we will use data collected from 5 drivers in 5 days, i.e. there will be 25 records. You can do anything to preprocess the data before feeding the data to the neural network, such as extracting features, getting sub-trajectory based on the status, and so on. This project should be completed in Python 3. Pytorch is highly recommended, but you can make your decision to use other tools like MxNet.

## Current Leaderboard
| rank | Name | Accuracy |
|---|---|---|
|**1**   | Shundong Li | 100% |
|**2**   | Saral Shrestha | 86% |
|**3**   | Vignesh Sundaram | 80% |
|**4**   | Padmesh Rajaram Naik | 80% |
|**5**   | Aviv Nur | 74% |
|**6**   | Mingjie Zeng | 72% |
|**7**   | Lichun Gao | 71% |
|**8**   | Seyed Bagher Hashemi Natanzi | 68% |
|**9**   | Anthony Chen | 65% |
|**10**   | Chao Wang | 63% |

## Deliverables & Grading
* PDF Report (50%) [template](https://www.acm.org/binaries/content/assets/publications/taps/acm_submission_template.docx)
    * proposal
    * methodology
    * empirical results and evaluation
    * conslusion
    
* Python Code (50%)
    * Code is required to avoid plagiarism.
    * The submission should contain a python file named "evaluation.py" to help evaluate your model. 
    * The evluation.py should follow the format in the Submission Guideline section. 
    * Evaluation criteria.
      | Percentage | Accuracy |
      |---|---|
      | 100 | 0.6 |
      | 90 | 0.55 |
      | 80 | 0.5 |
      | 70 | 0.45|
      | 60 | 0.4 |
* Grading:
  * Total (100):
    * Code (50) + Report (50)

  * Code (50):
    * accuracy >= 0.60: 50
    * accuracy >= 0.55: 45
    * accuracy >= 0.50: 40
    * accuracy >= 0.45: 35
    * accuracy >= 0.40: 30

  * Report (50):
    1. Introduction & Proposal (5)
    2. Methodology (20):
        a. Data processing (5)
        b. Feature generation (5)
        c. Network structure (5)
        d. Training & validation process (5)
    3. Evaluation & Results (20):
        a. Training & validation results (10)
        b. Performance comparing to your baselines (maybe different network structure) (5)
        c. Hyperparameter (learning rate, dropout, activation) (5)
    4. Conclusion (5)

  * Bonus (5):
   
     5 bonus points for the top 3 on the leader board.

## Project Guidelines

#### Dataset Description
| plate | longitute | latitude | time | status |
|---|---|---|---|---|
|4    |114.10437    |22.573433    |2016-07-02 0:08:45    |1|
|1    |114.179665    |22.558701    |2016-07-02 0:08:52    |1|
|0    |114.120682    |22.543751    |2016-07-02 0:08:51    |0|
|3    |113.93055    |22.545834    |2016-07-02 0:08:55    |0|
|4    |114.102051    |22.571966    |2016-07-02 0:09:01    |1|
|0    |114.12072    |22.543716    |2016-07-02 0:09:01    |0|


Above is an example of what the data look like. In the data/ folder, each .csv file is trajectories for 5 drivers in the same day. Data can be found at [Google Drive](https://drive.google.com/open?id=1xfyxupoE1C5z7w1Bn5oPRcLgtfon6xeT)
#### Feature Description 
* **Plate**: Plate means the taxi's plate. In this project, we change them to 0~5 to keep anonymity. Same plate means same driver, so this is the target label for the classification. 
* **Longitude**: The longitude of the taxi.
* **Latitude**: The latitude of the taxi.
* **Time**: Timestamp of the record.
* **Status**: 1 means taxi is occupied and 0 means a vacant taxi.

#### Problem Definition
Given a full-day trajectory of a taxi, you need to predict which taxi driver it belongs to. 

#### Evaluation 
Five days of trajectories will be used to evaluate your submission. And test trajectories are not in the data/ folder. 

#### Submission Guideline
To help better and fast evaluate your model, please submit a separate python file named "evaluation.py". This file should contain two functions.
* **Data Processing**
  ```python
  def processs_data(traj):
    """
    Input:
        Traj: a list of list, contains one trajectory for one driver 
        example:[[114.10437, 22.573433, '2016-07-02 00:08:45', 1],
           [114.179665, 22.558701, '2016-07-02 00:08:52', 1]]
    Output:
        Data: any format that can be consumed by your model.
    
    """
    return data
  ```
* **Model Prediction**
    ```python
    def run(data,model):
        """
        
        Input:
            Data: the output of process_data function.
            Model: your model.
        Output:
            prediction: the predicted label(plate) of the data, an int value.
        
        """
        return prediction
  ```

## Some Tips
Setup information could also be found in the [slides](https://docs.google.com/presentation/d/148pBkhw4HqGjkQGkOdsXjJw_B3rzVgi6Brq6fc7r8mE/edit?usp=sharing)
* Anaconda and virtual environment set tup
   * [Download and install anaconda](https://www.anaconda.com/distribution/)
   * [Create a virtual environment with commands](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
* Deep learning package
   * [Pytorch](https://pytorch.org/tutorials/)
   * [MxNet](https://mxnet.apache.org/)
* Open source GPU
   * [Turing](https://arc.wpi.edu/cluster-documentation/build/html/index.html)
   * [Using GPU on Google Cloud](https://github.com/yanhuata/DS504CS586-S20/blob/master/project2/keras_tutorial.ipynb)
   * [Cuda set up for Linux](https://docs.google.com/document/d/1rioVwqvZCbn58a_5wqs5aT3YbRsiPXs9KmIuYhmM1gY/edit?usp=sharing)
   * [Google colab](https://colab.research.google.com/notebooks/gpu.ipynb)
   * [Kaggle](https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu)
* **Keywords**. 
   * If you are wondering where to start, you can try to search "sequence classification", "sequence to sequence" or "sequence embedding" in Google or Github, this might provide you some insights.
   

