#!/usr/bin/env python3
'''
    @author Ryan
    @E-mail 497222072@qq.com
'''
import tensorflow as tf
import numpy as np
import os, argparse, time, random, sys
from model import BiLSTM_CRF
from utils import str2bool, get_logger, final_process
from data import read_corpus, read_dictionary, tag2label, random_embedding
from get_train_data import read_path_all
import datetime

##Path definition
TRAIN_PATH = "../data/train/" if os.path.isdir("../data/train/") else "../data/ruijin_round1_train2_20181022/"
print("Train path is {}".format(TRAIN_PATH))
TEST_PATH = "../data/test/" if os.path.isdir("../data/test") else "../data/ruijin_round1_test_b_20181112/"
print("Test path is {}".format(TEST_PATH))
SUBMIT_PATH = time_submit_dir = "../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+"/"    
## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory


## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data', help='train data source')
parser.add_argument('--test_data', type=str, default='data', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=20, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1542071314', help='model for test and demo')
args = parser.parse_args()

##pre-processing data
read_path_all(TRAIN_PATH, TEST_PATH)


## get char embeddings
word2id = read_dictionary(os.path.join('..', args.train_data, 'word2id.pkl'))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')


## read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join('..', args.train_data, 'train_data')
    test_path = os.path.join('..', args.test_data, 'test_data')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path); test_size = len(test_data)


## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
the_output_path = os.path.join('..', 'data',"model_save")
if not os.path.isdir(the_output_path):
    os.mkdir(the_output_path)
output_path = os.path.join(the_output_path, timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


## training model
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    ## hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    ## train model on the whole training data
    print('============= Training =============')
    print("train data: {}".format(len(train_data)))
    with tf.Session(config=config) as sess:
        model.train(train=train_data, dev=test_data, sess=sess)  # use test_data as the dev_data to see overfitting phenomena

        ## testing model
        # elif args.mode == 'test':
        # ckpt_file = tf.train.latest_checkpoint(model_path)
        # print(ckpt_file)
        # paths['model_path'] = ckpt_file
        # model_test = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
        # model_test.build_graph()
        # print("test data: {}".format(test_size))
        # model.test(test_data)

        ruijin_test_path = TEST_PATH
        test_file_name_list = os.listdir(ruijin_test_path)
        # saver = tf.train.Saver()
        # with tf.Session(config=config) as sess:
        print('============= Testing =============')
        # saver.restore(sess, ckpt_file)    
        if not os.path.isdir(SUBMIT_PATH):
            os.mkdir(SUBMIT_PATH)
        print("submit path is {}".format(SUBMIT_PATH))
        for ff in test_file_name_list:
            file_name = ff.split(".")
            print("generating ann file: {}".format(file_name[0]+".ann"))
            with open(ruijin_test_path + ff, "r")as f:
                sent = f.readlines()
            new_sent = ""
            origin_list = []
            flag_list = []
            n_list = []
            for s in sent:
                # new_sent += s.strip()
                for w in s:
                    if not w.strip():
                        if w == "\n":
                            n_list.append(len(origin_list))
                        else:
                            flag_list.append(len(origin_list))
                    origin_list.append(w)
            for word in origin_list:
                new_sent += word.strip()

            sent_data = [(new_sent, ['O'] * len(new_sent))]
            tag = model.demo_one(sess, sent_data)
            

            result_path = SUBMIT_PATH + file_name[0] + ".ann"
            
            # with open(result_path, "w")as fout:
                # fout.write(str(tag))
            final_process(result_path, origin_list, flag_list, n_list, tag)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while(1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                
                # print(get_entity(tag, demo_sent))
                # PER, LOC, ORG = get_entity(tag, demo_sent)
                # print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
