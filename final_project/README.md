# MOOC Dropout Prediction
Final project of Machine Learning course

Achieve 0.968 of [MAP](https://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision)@9000

## Problem Description
The Data are from a Massive Open Online Course (MOOC) platform,
which contains information of many courses and students. Your goal is to predict whether a student will
drop a course that she/he enrolled in. We provide you all the logs of each enrollment within the first
30 days of the course. Then, if the student has no logs within the following 10 days, i.e. the 31st-40th
days from the start date of the course, we label him/her as a **dropout**. (Note that the 31st day starts
from 00:00:00). *Data is originated from KDDCup 2015, which aims for a similar goal with a
different evaluation criterion*

## Data Description
 1. Enrollment: match the enrollment id to student and course.
 2. Log: Logs for each enrollment (including event type and time). (~8,000,000 rows)
 3. Object: Contain information about courses. Each course is represented as a tree of modules.
For instance, a course contains multiple chapter modules, a chapter contains sequentials, and a
sequential contains verticals and videos.
 4. We have total 96,434 of "labeled" training data and 24,109 testing data

## Our Methods
### Model
We have tested with Logistic Regression, Support Vector Machine, Random Forest and Gradient Boosted Decision Tree.

GBDT gives the best result overall.

### Features
This key to this project is **feature engineering**. We have extracted total 482 features. Below I list a few of what we've extracted.
1. Basic Features:
    - total number of logs of the user (student) in all the courses
    - total number of logs belongs to the course
    - number of courses the user takes
    - number of users who take the course
    - number of logs belongs to the enrollment
    - (event source) (event type): 9 dimensions, number of logs with different event sources and
event types (refer to log train/test.csv)
    - (chapter/sequentail/video) count: 3 dimensions, number of logs with certain objects
2. Time Related Features:
    - number of days between user enrollment date and the course start date
    - duration of logs
    - number of days user haven't use the platform since last time
    - the enroll start day of course 
3. Frequency Related Features:
    - the average number of logs in Monday, Tuesday... of each week
4. Extend Features:
    - for the same user, if he/she is enrolled in multiple course and the course log is overlapped,
    then we can user other course's log to better guess if he/she has dropped the course or not
5. Course Similarity Features

## Evaluation
For details refer to [spec](https://github.com/YuanTingHsieh/ML_2015/blob/master/final_project/spec.pdf)
1. Mean Average Precision (MAP) @ 9000
2. Weighted Accuracy
