操作系统：ubuntu 16.04 LTS 64bit
语言：Python3.5
Main Package:
    tensorflow GPU 1.12.0 (CUDA 9.0 CUDNN 7.3.1) 
    numpy 1.15.4
    pickle
    os, random, sys....
程序运行：python3 main.py

./data/
    测试文件路径： ruijin_round1_test_b_20181112/*.txt （请务必放入ruijin_round1_test_b_20181112整个文件夹）
    训练文件路径： ruijin_round1_train2_20181022/*.ann, *.txt  （请务必放入ruijin_round1_train2_20181022整个文件夹）
    模型保存路径： model_save/
    BIO中间文件：train_data, test_data, word2id.pkl

./code/
    main.py
    data.py
    model.py
    eval.py
    .......

./submit/
    结果文件路径: submit_20180203_040506/*.ann
