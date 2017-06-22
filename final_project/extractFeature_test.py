import pandas as pd
import numpy as np
from datetime import datetime as dt

with open("enrollment_test.csv") as f:
  enrollment = pd.read_csv(f)
with open("log_test.csv") as f:
  log = pd.read_csv(f)
with open("object.csv") as f:
  courseware = pd.read_csv(f, delimiter=',')
with open("sample_test_x.csv") as f:
  sample = pd.read_csv(f)

feature = pd.DataFrame({"enrollment_id" : enrollment.enrollment_id})
doneStudentId = []
allCourseId = courseware.course_id.unique()
courseModuleNum = pd.DataFrame({"course_id" : allCourseId})
for courseId in courseModuleNum.course_id:
  courseModuleNum.loc[courseModuleNum.course_id == courseId, "module_num"] = len(courseware[courseware.course_id == courseId].index)

for enrollmentId in feature.enrollment_id:
  time = [dt.strptime(time, "%Y-%m-%dT%H:%M:%S") for time in log[log.enrollment_id == enrollmentId].time]
  lastLogTime = max(time)
  duration = lastLogTime - min(time)
  feature.loc[feature.enrollment_id == enrollmentId, "duration"] = 86400 * duration.days + duration.seconds

  interval = np.diff(time)
  interval = [i.days for i in interval if i.days != 0]
  #interval = [86400 * i.days + i.seconds for i in interval]
  feature.loc[feature.enrollment_id == enrollmentId, "interval_max"] = max(interval) if interval else 0
  feature.loc[feature.enrollment_id == enrollmentId, "interval_min"] = min(interval) if interval else 0

  courseId = enrollment.loc[enrollment.enrollment_id == enrollmentId, "course_id"].iloc[0]
  courseTime = [dt.strptime(time, "%Y-%m-%dT%H:%M:%S") for time in courseware[courseware.course_id == courseId].start if time != "null"]
  idle = max(courseTime) - lastLogTime
  feature.loc[feature.enrollment_id == enrollmentId, "idle"] = 86400 * idle.days + idle.seconds

  studentId = enrollment.loc[enrollment.enrollment_id == enrollmentId, "username"].iloc[0]
  if studentId not in doneStudentId:
    allHisEnrollment = enrollment[enrollment.username == studentId]
    logNumList = []
    logNumRatioList = []
    for courseId, courseNo in zip(allCourseId, range(len(allCourseId))):
      name = "log_num_in_course_" + str(courseNo)
      if courseId in allHisEnrollment.course_id.values:
        tempEnrollmentId = allHisEnrollment.loc[allHisEnrollment.course_id == courseId, "enrollment_id"].iloc[0]
        logNum = len(log[log.enrollment_id == tempEnrollmentId].index)
        logNumList.append(logNum)
      else:
        logNum = 0
      for hisEnrollmentId in allHisEnrollment.enrollment_id:
        feature.loc[feature.enrollment_id == hisEnrollmentId, name] = logNum
      name = "log_num_ratio_in_course_" + str(courseNo)
      moduleNum = courseModuleNum.loc[courseModuleNum.course_id == courseId, "module_num"].iloc[0]
      logNumRatio = logNum / float(moduleNum)
      if logNum != 0:
        logNumRatioList.append(logNumRatio)
      for hisEnrollmentId in allHisEnrollment.enrollment_id:
        feature.loc[feature.enrollment_id == hisEnrollmentId, name] = logNumRatio
    for hisEnrollmentId in allHisEnrollment.enrollment_id:
      feature.loc[feature.enrollment_id == hisEnrollmentId, "log_num_in_course_mean"] = np.mean(logNumList)
      feature.loc[feature.enrollment_id == hisEnrollmentId, "log_num_in_course_std"] = np.std(logNumList)
      feature.loc[feature.enrollment_id == hisEnrollmentId, "log_num_ratio_in_course_mean"] = np.mean(logNumRatioList)
      feature.loc[feature.enrollment_id == hisEnrollmentId, "log_num_ratio_in_course_std"] = np.std(logNumRatioList)
    doneStudentId.append(studentId)

  logSum = sum(sample.loc[sample.ID == enrollmentId].iloc[0, 6:15])
  for name in sample.columns[6:15]:
    newFeatName = name + "_ratio"
    feature.loc[feature.enrollment_id == enrollmentId, newFeatName] = sample.loc[sample.ID == enrollmentId, name] / float(logSum)

with open("feature_test_chichi.csv", "w") as f:
  feature.to_csv(f, index = False)
