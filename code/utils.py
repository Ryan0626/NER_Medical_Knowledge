'''
    @Tools
    @author Ryan
    @E-mail 497222072@qq.com
'''

import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def final_process(result_path, origin_list, flag_list, n_list, tag):
    entity_cnt = 1
    with open(result_path, "w")as fout:
        cnt = 0
        i = 0
        slot = [False, "", [], ""]
        slot_list = []
        tmp = []
        cnt = 0
        for i in range(len(origin_list)):
            if i not in flag_list and i not in n_list:
                tmp.append([origin_list[i], str(tag[cnt])])
                cnt += 1
            elif i in n_list:
                tmp.append([origin_list[i], "N"])
            else:
                tmp.append([origin_list[i], "Z"])
        
        for j in range(len(tmp)):
            class_name = tmp[j][1].split("-")
            
            if class_name[0] == "B":
                real_tmp = [[], ""]
                if slot[0] == False:
                    slot[0] = True
                    slot[1] = class_name[1]
                    slot[2].append(j)
                    slot[3] += tmp[j][0]
                else:
                    slot_list.append(slot)
                    slot = [True, "", [], ""]
                    slot[1] = class_name[1]
                    slot[2].append(j)
                    slot[3] += tmp[j][0]
            elif class_name[0] == "I":
                # if slot[0] == False:
                #     print("Error")
                # else:
                if slot[0] == True:
                    slot[2] += real_tmp[0]
                    slot[3] += real_tmp[1]
                    slot[2].append(j)
                    slot[3] += tmp[j][0]
                    real_tmp = [[], ""]
            elif class_name[0] == "N":
                if slot[0] == True:
                    real_tmp[0].append("N")
                    real_tmp[1] += " "
            elif class_name[0] == "Z":
                if slot[0] == True:
                    real_tmp[0].append("Z")
                    real_tmp[1] += " "

            elif class_name[0] =="0":
                if slot[0] == True:
                    real_tmp[0].append(j)
                    real_tmp[1] += tmp[j][0]


        for k in range(len(slot_list)):
            fout.write("T" + str(k) +"\t")
            fout.write(slot_list[k][1] + " ")
            l = len(slot_list[k][2])
            i = 0
            final_sy = []
            
            while i < l:
                sy = []
                while i < l and slot_list[k][2][i] != "N":
                    sy.append(slot_list[k][2][i])
                    i += 1
                if sy != []:
                    final_sy.append(sy)
                    sy = []
                i += 1
            
            fout.write(str(final_sy[0][0]) + " " +str(final_sy[0][-1]+1))
            if len(final_sy) > 1:
                # if len(final_sy) > 2:
                #     print("lookkkkkkkkkkkkkkkkkkkkkkk", result_path)
                for h in final_sy[1:]:           
                    fout.write(";" + str(h[0]) + " " + str(h[-1] + 1))
            fout.write("\t" + slot_list[k][3] + "\n")

