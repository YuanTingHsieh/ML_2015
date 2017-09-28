import pandas as pd
import numpy as np


with open("enrollment_train.csv") as f:
  enrollment_train = pd.read_csv(f)

with open("enrollment_test.csv") as f:
  enrollment_test = pd.read_csv(f)

with open("sample_test_x.csv") as f:
  sample_test = pd.read_csv(f)

with open("sample_train_x.csv") as f:
  sample_train = pd.read_csv(f)

with open("object.csv") as f:
  courseware = pd.read_csv(f, delimiter=',')

allCourseId = courseware.course_id.unique()
courseModuleNum = pd.DataFrame({"course_id" : allCourseId})

all_sample_t = [sample_train,sample_test]
all_sample = pd.concat(all_sample_t)


all_enroll_t = [enrollment_train,enrollment_test]
all_enroll = pd.concat(all_enroll_t)

allUser = all_enroll.username.unique()
#print type(allUser)
#print len(allUser)
np.zeros((len(allUser),len(allCourseId)),dtype=np.int)
theMatrix = pd.DataFrame(np.zeros((len(allUser),len(allCourseId)),dtype=np.int) , index=allUser, columns=allCourseId)
#print theMatrix[allCourseId[1]][allUser[1] ]
for enrollId in all_enroll['enrollment_id'].values:
  print enrollId
  #print type(allUser[1])
  #print type(all_sample[all_sample['ID']==enrollId].log_num.values[0])
  theMatrix[all_enroll[all_enroll['enrollment_id']==enrollId].course_id.values[0] ][all_enroll[all_enroll['enrollment_id']==enrollId].username.values[0] ] = all_sample[all_sample['ID']==enrollId].log_num.values[0]

with open("theMatrix.csv", "w") as f:
  theMatrix.to_csv(f, index = False)
