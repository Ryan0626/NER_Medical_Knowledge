'''
    @pre-processing data
    @author Ryan
    @E-mail 497222072@qq.com
'''

import collections
import os
import pickle


OUTPUT_PATH = "../data/"
count_dic = {}

def origin_data_reader(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    data = []
    data_str = ''
    for line in lines:
        data_str += line
    for d in data_str:
        data.append(d)

    return data

def read_file(src_file):
    '''
    :return: list of src
    '''
    with open(src_file, "r") as f:
        src_lines = f.readlines()

    src_list = []
    for line in src_lines:
        src_list.append(line.split())
    return src_list

def merge_label(ann_data, origin_data):
    
    data = origin_data    
    l = len(data)
    tmp = [0] * l
    for i in ann_data:
        n = int(i[2])
        if tmp[n] == 0:
            data[n] += "\tB-" + i[1]
            if i[1] in count_dic:
                count_dic[i[1]] += 1
            else:
                count_dic[i[1]] = 1
            tmp[n] = 1
            j = 3
            while ";" in i[j]:
                special_format = i[j].split(";")
                m = int(special_format[0])
                data[m] = " "
                for k in range(n+1, m):                 
                    if data[k].strip() and tmp[k] == 0:
                        data[k] += "\tI-" + i[1]         
                        tmp[k] = 1
                n = int(special_format[1]) - 1
                j += 1
            m = int(i[j])
            for a in range(n+1, m):
                # print(a)
                
                if data[a].strip() and tmp[a] == 0:
                    data[a] += "\tI-" + i[1]
                    tmp[a] = 1

    for i in range(l):
        if data[i].strip():
            if tmp[i] == 1:
                data[i] += "\n"
            else:
                data[i] += "\tO\n" 
    return data

def read_path_all(TRAIN_PATH, TEST_PATH):
    print("Pre-processing data...")
    cnt = 0
    train_result = []
    test_result = []
    word_collection = []
    files = os.listdir(TRAIN_PATH)
    l = len(files)
    for f in files:
        file_name = f.split(".")
        if cnt != 10:
            if file_name[1] == "txt":
                cnt += 1
                # print("train", file_name[0])
                origin_data = origin_data_reader(TRAIN_PATH+f)
                word_collection += origin_data
                ann_data = read_file(TRAIN_PATH+file_name[0]+".ann")
                labeled_data = merge_label(ann_data, origin_data)
                train_result += labeled_data
        else:
            if file_name[1] == "txt":
                cnt = 0
                # print("test", file_name[0])
                origin_data = origin_data_reader(TRAIN_PATH+f)
                word_collection += origin_data
                ann_data = read_file(TRAIN_PATH+file_name[0]+".ann")
                labeled_data = merge_label(ann_data, origin_data)
                test_result += labeled_data
            
    real_test_files = os.listdir(TEST_PATH)
    for tst_f in real_test_files:
        test_f_ = TEST_PATH + tst_f 
        origin_test = origin_data_reader(test_f_)
        word_collection += origin_test

            
    with open(OUTPUT_PATH+"train_data", "w") as f:
        for i in train_result:
            if i.strip():
                f.write(i.strip()+"\n")
            elif i == "\n":
                f.write(i)
            
    with open(OUTPUT_PATH+"test_data", "w") as f:
        for i in test_result:
            if i.strip():
                f.write(i.strip()+"\n")
            elif i == "\n":
                f.write(i) 


    counter = collections.Counter(word_collection)

    counter_pairs = sorted(counter.items(), key=lambda x: ( -x[1], x[0]))
    words, _ = list(zip(*counter_pairs))
    
    NEW = ("<UNK>", "<ENG>", "<NUM>")
    words += NEW
    # print(len(words))
    word2id = dict(zip(words, range(len(words))))
    with open(OUTPUT_PATH+"word2id.pkl", "wb") as f:
        pickle.dump(word2id, f)