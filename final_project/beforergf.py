import numpy as np
import myparse as mp
from sklearn.preprocessing import StandardScaler, LabelEncoder


# read csv include first line
enroll_train = mp.readcsv("enrollment_train.csv")
enroll_test = mp.readcsv("enrollment_test.csv")

truth_train = mp.readcsv("truth_train.csv")
sample_train_x = mp.readcsv("sample_train_x.csv")
sample_test_x = mp.readcsv("sample_test_x.csv")
aug_graph_train = mp.readcsv("augmentGraph_train.csv")
aug_graph_test =mp.readcsv("augmentGraph_test.csv")

all_feat_train = mp.readcsv("feat_train.csv")
all_feat_test = mp.readcsv("feat_test.csv")

all_azure_train = mp.readcsv("azure_train.csv")
all_azure_test = mp.readcsv("azure_test.csv")

all_azure2_train = mp.readcsv("azure2_train.csv")
all_azure2_test = mp.readcsv("azure2_test.csv")

all_chichi_train = mp.readcsv("feature_train_chichi.csv")
all_chichi_test = mp.readcsv("feature_test_chichi.csv")

all_svd_train = mp.readcsv("newfeatwithsvd0111_train.csv")
all_svd_test = mp.readcsv("newfeatwithsvd0111_test.csv")

all_bycourse_train = mp.readcsv("bycoursefeat_train.csv")
all_bycourse_test = mp.readcsv("bycoursefeat_test.csv")

#pseudo1_train = mp.readcsv("XGB_5_005_08_07_300_all_0111_mapout_train.csv")
#pseudo1_test = mp.readcsv("XGB_5_005_08_07_300_all_0111_mapout_test.csv")

all_users = np.append(enroll_train[1:,1],enroll_test[1:,1])
all_courses = np.append(enroll_train[1:,2],enroll_test[1:,2])


#Start Encode Courses and Users
llle = LabelEncoder()

users_id = llle.fit_transform(all_users).astype(float)
courses_id = llle.fit_transform(all_courses).astype(float)

train_users_id = users_id[0:len(sample_train_x)-1]
test_users_id = users_id[(len(sample_train_x)-1):]
train_courses_id = courses_id[0:len(sample_train_x)-1]
test_courses_id = courses_id[(len(sample_train_x)-1):]



#Segmenting data to use
data_train = sample_train_x[1:,1:].astype(float)
aug_train = aug_graph_train[1:,1:].astype(float)
feat_train = all_feat_train[0:,1:].astype(float)
azure_train = all_azure_train[1:,2:].astype(float)
azure2_train = all_azure2_train[1:,2:33].astype(float)
chichi_train = all_chichi_train[1:,1:].astype(float)
bycourse_train = all_bycourse_train[1:,1].astype(float)
svd_train = all_svd_train[0:,0:].astype(float)
#ps_train = pseudo1_train[0:,1].astype(float)

data_test = sample_test_x[1:,1:].astype(float)
aug_test = aug_graph_test[1:,1:].astype(float)
feat_test = all_feat_test[0:,1:].astype(float)
azure_test = all_azure_test[1:,2:].astype(float)
azure2_test = all_azure2_test[1:,2:33].astype(float)
chichi_test = all_chichi_test[1:,1:].astype(float)
bycourse_test = all_bycourse_test[1:,1].astype(float)
svd_test = all_svd_test[0:,0:].astype(float)
#ps_test = pseudo1_test[0:,1].astype(float)

data_train = np.hstack((data_train,aug_train))
data_train = np.hstack((data_train,feat_train))
data_train = np.hstack((data_train,azure_train))
data_train = np.hstack((data_train,azure2_train))
data_train = np.hstack((data_train,chichi_train))
data_train = np.hstack((data_train,svd_train))

label_train = truth_train[0:,1].astype(float)

data_test = np.hstack((data_test,aug_test))
data_test = np.hstack((data_test,feat_test))
data_test = np.hstack((data_test,azure_test))
data_test = np.hstack((data_test,azure2_test))
data_test = np.hstack((data_test,chichi_test))
data_test = np.hstack((data_test,svd_test))

#Preprocessing
preprocess = StandardScaler()
preprocess.fit(np.vstack((data_train,data_test)))
data_train = preprocess.transform(data_train)
data_test = preprocess.transform(data_test)

#Adding User ID and Course ID
fuck_train = np.vstack((train_users_id,train_courses_id))
#fuck_train = np.vstack((fuck_train,bycourse_train))
#fuck_train = np.vstack((fuck_train,ps_train))
data_train = np.hstack((np.transpose(fuck_train),data_train))
print np.shape(data_train)

fuck_test = np.vstack((test_users_id,test_courses_id))
#fuck_test = np.vstack((fuck_test,bycourse_test))
#fuck_test = np.vstack((fuck_test,ps_test))
data_test = np.hstack((np.transpose(fuck_test),data_test))

label_train[label_train==0]=-1

np.savetxt("data_train_forrgf", data_train, delimiter=" ",fmt="%s")
np.savetxt("data_test_forrgf", data_test, delimiter=" ",fmt="%s")

np.savetxt("label_train_forrgf", np.c_[label_train],delimiter=" ",fmt="%s")