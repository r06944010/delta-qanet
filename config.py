import os
import tensorflow as tf
import sys
'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

from prepro_char_kgb import prepro
from main import train, test, demo

flags = tf.flags

home = os.path.expanduser("~")

if sys.argv[2] == 'TOCFL':
    train_dir = "train_tocfl_" + str(sys.argv[1])
    target_dir = "data_tocfl_" + str(sys.argv[1])
    train_file = None
    dev_file = None
    test_file = os.path.join(home, "corpus", "tocfl", "transcription.csv")
    glove_word_file = None

elif sys.argv[2] == 'Delta':
    train_file = os.path.join(home, "corpus", "DRCD", "DRCD_training.json")
    dev_file = os.path.join(home, "corpus", "DRCD", "DRCD_dev.json")
    test_file = os.path.join(home, "corpus", "DRCD", "DRCD_dev.json")
    # glove_word_file = os.path.join(home, "corpus", "glove", "glove.840B.300d.txt")
    glove_word_file = None
    train_dir = "train_delta_" + str(sys.argv[1])
    target_dir = "data_kgb_" + str(sys.argv[1])
    
elif sys.argv[2] == 'KGB':
    train_file = None
    dev_file = None
    glove_word_file = None
    test_file = os.path.join("kgb", "0622.json")
    train_dir = "train_delta_" + str(sys.argv[1])
    target_dir = "data_kgb_" + str(sys.argv[1])

# model_name = sys.argv[1] # "all"
dir_name = os.path.join(train_dir)

if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(os.path.join(os.getcwd(),dir_name)):
    os.mkdir(os.path.join(os.getcwd(),dir_name))

embedding_dir = "data_kgb_" + str(sys.argv[1])

log_dir = os.path.join(dir_name, "event")
save_dir = os.path.join(dir_name, "model")
answer_dir = os.path.join(dir_name, "answer")
train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")

word_emb_file = os.path.join(embedding_dir, "word_emb.json")
char_emb_file = os.path.join(embedding_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")

word_dictionary = os.path.join(embedding_dir, "word_dictionary.json")
char_dictionary = os.path.join(embedding_dir, "char_dictionary.json")
answer_file = os.path.join('kgb', "kgb_answer.json")
answer_csv = os.path.join('kgb', "kgb_answer.csv")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)

flags.DEFINE_string("mode", "train", "Running mode train/debug/test")

flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")
flags.DEFINE_string("train_file", train_file, "Train source file")
flags.DEFINE_string("dev_file", dev_file, "Dev source file")
flags.DEFINE_string("test_file", test_file, "Test source file")
flags.DEFINE_string("glove_word_file", glove_word_file, "Glove word embedding source file")

flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")
flags.DEFINE_string("dev_record_file", dev_record_file, "Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")
flags.DEFINE_string("char_emb_file", char_emb_file, "Out file for char embedding")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")
flags.DEFINE_string("answer_file", answer_file, "Out file for answer")
flags.DEFINE_string("answer_csv", answer_csv, "Out file for answer")
flags.DEFINE_string("word_dictionary", word_dictionary, "Word dictionary")
flags.DEFINE_string("char_dictionary", char_dictionary, "Character dictionary")


flags.DEFINE_integer("glove_char_size", 94, "Corpus size for Glove")
flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 64, "Embedding dimension for char")

flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("ans_limit", 30, "Limit length for answers")
flags.DEFINE_integer("test_para_limit", 1750, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 200, "Limit length for question in test file")
flags.DEFINE_integer("test_ans_limit", 100, "Limit length for answer in test file")
flags.DEFINE_integer("char_limit", 16, "Limit length for character")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
flags.DEFINE_list("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("num_steps", 60000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("dropout", 0.1, "Dropout prob across the layers")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("decay", 0.9999, "Exponential moving average decay")
flags.DEFINE_float("l2_norm", 3e-7, "L2 norm scale")
flags.DEFINE_integer("hidden", 96, "Hidden size")
flags.DEFINE_integer("num_heads", 1, "Number of heads in self attention")
flags.DEFINE_integer("early_stop", 10, "Checkpoints for early stop")

# Extensions (Uncomment corresponding code in download.sh to download the required data)
glove_char_file = os.path.join(home, "data", "glove", "glove.840B.300d-char.txt")
flags.DEFINE_string("glove_char_file", glove_char_file, "Glove character embedding source file")
flags.DEFINE_boolean("pretrained_char", False, "Whether to use pretrained character embedding")

fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding source file")
flags.DEFINE_boolean("fasttext", False, "Whether to use fasttext")

# Self define CHINESE word to vector
tw_w2v = os.path.join(home, "corpus", "advdl", "word2vec", "DRCD_w2v_300.txt")
tw_c2v = os.path.join(home, "corpus", "advdl", "word2vec", "single_w2v_300.txt")

flags.DEFINE_string("tw_w2v", tw_w2v, "ta word embedding chinese")
flags.DEFINE_string("tw_c2v", tw_c2v, "ta character embedding chinese")

flags.DEFINE_integer("tw_word_size", 113414, "Corpus size for Glove")
flags.DEFINE_integer("tw_char_size", 174894, "Corpus size for char2vec")
flags.DEFINE_integer("tw_word_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("tw_char_dim", 300, "Embedding dimension for char")

# flags.DEFINE_string("type", "all", "word/char embedding choose")
flags.DEFINE_string("type", str(sys.argv[1]).split('_')[0], "word/char embedding choose")

def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        prepro(config)
    elif config.mode == "debug":
        config.num_steps = 2
        config.val_num_batches = 1
        config.checkpoint = 1
        config.period = 1
        train(config)
    elif config.mode == "test":
        test(config)
    elif config.mode == "demo":
        demo(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
