#!/usr/bin/env python
# -*- coding: utf-8 -*-
import  numpy as np
import  csv


def readcsv(input_file):
    print "Reading "+input_file
    with open(input_file,'rb') as csvfile:
        data_iter = csv.reader(csvfile,delimiter=',')
        data = [data for data in data_iter]
    data_array = np.asarray(data)
    return data_array

def writecsv(output_file, dic_to_dump):
    print "Writing "+output_file
    with open(output_file,'wb') as csvfile:
        fieldnames=dic_to_dump[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dic_to_dump)


#readcsv("object.csv")
#readcsv("log_train.csv")
#readcsv("sample_train_x.csv")

