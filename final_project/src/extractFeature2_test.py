import pandas as pd
import numpy as np
from datetime import datetime as dt

with open("enrollment_test.csv") as f:
  enrollment = pd.read_csv(f)
with open("log_test.csv") as f:
  log = pd.read_csv(f)
with open("feature_test_chichi.csv") as f:
  chichiFeature = pd.read_csv(f)

feature = pd.DataFrame({"enrollment_id" : enrollment.enrollment_id})
doneStudentId = set()
log.time = [dt.strptime(time, "%Y-%m-%dT%H:%M:%S") for time in log.time]

for enrollmentId in feature.enrollment_id:
  studentId = enrollment.loc[enrollment.enrollment_id == enrollmentId, "username"].iloc[0]
  if studentId not in doneStudentId:
    allHisEnrollment = enrollment[enrollment.username == studentId]
    dropNum7days = len(chichiFeature[(chichiFeature.enrollment_id == enrollmentId) \
                                      & (chichiFeature.duration < 23 * 86400)].index)
    dropNum10days = len(chichiFeature[(chichiFeature.enrollment_id == enrollmentId) \
	                                  & (chichiFeature.duration < 20 * 86400)].index)
    for hisEnrollmentId in allHisEnrollment.enrollment_id:
      name = "drop_course_num_7days"
      feature.loc[feature.enrollment_id == hisEnrollmentId, name] = dropNum7days
      name = "drop_course_num_10days"
      feature.loc[feature.enrollment_id == hisEnrollmentId, name] = dropNum10days
    doneStudentId.add(studentId)

  firstLogTime = min(log[log.enrollment_id == enrollmentId].time)
  dayNo = [(time - firstLogTime).days + 1 for time in log[log.enrollment_id == enrollmentId].time]
  log.loc[log.enrollment_id == enrollmentId, "day_No"] = dayNo
  for i in range(30):
    name = "server_nagivate_num_day" + str(i + 1)
    logNum = len(log[(log.enrollment_id == enrollmentId) \
	                 & (log.source == "server") \
                     & (log.event == "nagivate")
                     & (log.day_No == i + 1)].index)
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(30):
    name = "server_access_num_day" + str(i + 1)
    logNum = len(log[(log.enrollment_id == enrollmentId) \
	                 & (log.source == "server") \
                     & (log.event == "access")
                     & (log.day_No == i + 1)].index)
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(30):
    name = "server_problem_num_day" + str(i + 1)
    logNum = len(log[(log.enrollment_id == enrollmentId) \
	                 & (log.source == "server") \
                     & (log.event == "problem")
                     & (log.day_No == i + 1)].index)
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(30):
    name = "browser_access_num_day" + str(i + 1)
    logNum = len(log[(log.enrollment_id == enrollmentId) \
	                 & (log.source == "browser") \
                     & (log.event == "access")
                     & (log.day_No == i + 1)].index)
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(30):
    name = "browser_problem_num_day" + str(i + 1)
    logNum = len(log[(log.enrollment_id == enrollmentId) \
	                 & (log.source == "browser") \
                     & (log.event == "problem")
                     & (log.day_No == i + 1)].index)
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(30):
    name = "browser_page_close_num_day" + str(i + 1)
    logNum = len(log[(log.enrollment_id == enrollmentId) \
	                 & (log.source == "browser") \
                     & (log.event == "page_close")
                     & (log.day_No == i + 1)].index)
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(30):
    name = "browser_video_num_day" + str(i + 1)
    logNum = len(log[(log.enrollment_id == enrollmentId) \
	                 & (log.source == "browser") \
                     & (log.event == "video")
                     & (log.day_No == i + 1)].index)
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(30):
    name = "server_discussion_num_day" + str(i + 1)
    logNum = len(log[(log.enrollment_id == enrollmentId) \
	                 & (log.source == "server") \
                     & (log.event == "discussion")
                     & (log.day_No == i + 1)].index)
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(30):
    name = "server_wiki_num_day" + str(i + 1)
    logNum = len(log[(log.enrollment_id == enrollmentId) \
	                 & (log.source == "server") \
                     & (log.event == "wiki")
                     & (log.day_No == i + 1)].index)
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum

  for i in range(4):
    name = "server_nagivate_num_week" + str(i + 1)
    logNum = sum(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, (7 * i + 3):(7 * i + 10)])
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(4):
    name = "server_access_num_week" + str(i + 1)
    logNum = sum(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, (7 * i + 33):(7 * i + 40)])
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(4):
    name = "server_problem_num_week" + str(i + 1)
    logNum = sum(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, (7 * i + 63):(7 * i + 70)])
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(4):
    name = "browser_access_num_week" + str(i + 1)
    logNum = sum(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, (7 * i + 93):(7 * i + 100)])
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(4):
    name = "browser_problem_num_week" + str(i + 1)
    logNum = sum(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, (7 * i + 123):(7 * i + 130)])
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(4):
    name = "browser_page_close_num_week" + str(i + 1)
    logNum = sum(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, (7 * i + 153):(7 * i + 160)])
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(4):
    name = "browser_video_num_week" + str(i + 1)
    logNum = sum(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, (7 * i + 183):(7 * i + 190)])
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(4):
    name = "server_discussion_num_week" + str(i + 1)
    logNum = sum(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, (7 * i + 213):(7 * i + 220)])
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum
  for i in range(4):
    name = "server_wiki_num_week" + str(i + 1)
    logNum = sum(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, (7 * i + 243):(7 * i + 250)])
    feature.loc[feature.enrollment_id == enrollmentId, name] = logNum

  log1Diff = np.diff(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, 273:277])
  print(len(log1Diff))
  for i in range(3):
    name = "server_nagivate_1diff_week" + str(i + 2)
    feature.loc[feature.enrollment_id == enrollmentId, name] = log1Diff[i]
  name = "server_nagivate_drop"
  feature.loc[feature.enrollment_id == enrollmentId, name] = min(log1Diff)
  log1Diff = np.diff(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, 277:281])
  for i in range(3):
    name = "server_access_1diff_week" + str(i + 2)
    feature.loc[feature.enrollment_id == enrollmentId, name] = log1Diff[i]
  name = "server_access_drop"
  feature.loc[feature.enrollment_id == enrollmentId, name] = min(log1Diff)
  log1Diff = np.diff(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, 281:285])
  for i in range(3):
    name = "server_problem_1diff_week" + str(i + 2)
    feature.loc[feature.enrollment_id == enrollmentId, name] = log1Diff[i]
  name = "server_problem_drop"
  feature.loc[feature.enrollment_id == enrollmentId, name] = min(log1Diff)
  log1Diff = np.diff(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, 285:289])
  for i in range(3):
    name = "browser_access_1diff_week" + str(i + 2)
    feature.loc[feature.enrollment_id == enrollmentId, name] = log1Diff[i]
  name = "browser_access_drop"
  feature.loc[feature.enrollment_id == enrollmentId, name] = min(log1Diff)
  log1Diff = np.diff(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, 289:293])
  for i in range(3):
    name = "browser_problem_1diff_week" + str(i + 2)
    feature.loc[feature.enrollment_id == enrollmentId, name] = log1Diff[i]
  name = "browser_problem_drop"
  feature.loc[feature.enrollment_id == enrollmentId, name] = min(log1Diff)
  log1Diff = np.diff(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, 293:297])
  for i in range(3):
    name = "browser_page_1diff_week" + str(i + 2)
    feature.loc[feature.enrollment_id == enrollmentId, name] = log1Diff[i]
  name = "browser_page_drop"
  feature.loc[feature.enrollment_id == enrollmentId, name] = min(log1Diff)
  log1Diff = np.diff(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, 297:301])
  for i in range(3):
    name = "browser_video_1diff_week" + str(i + 2)
    feature.loc[feature.enrollment_id == enrollmentId, name] = log1Diff[i]
  name = "browser_video_drop"
  feature.loc[feature.enrollment_id == enrollmentId, name] = min(log1Diff)
  log1Diff = np.diff(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, 301:305])
  for i in range(3):
    name = "server_discussion_1diff_week" + str(i + 2)
    feature.loc[feature.enrollment_id == enrollmentId, name] = log1Diff[i]
  name = "server_discussion_drop"
  feature.loc[feature.enrollment_id == enrollmentId, name] = min(log1Diff)
  log1Diff = np.diff(feature.loc[feature.enrollment_id == enrollmentId].iloc[0, 305:309])
  for i in range(3):
    name = "server_wiki_1diff_week" + str(i + 2)
    feature.loc[feature.enrollment_id == enrollmentId, name] = log1Diff[i]
  name = "server_wiki_drop"
  feature.loc[feature.enrollment_id == enrollmentId, name] = min(log1Diff)

with open("feature2_test_chichi.csv", "w") as f:
  feature.to_csv(f, index = False)
