import random,json,re,itertools,copy,logging
import tensorflow as tf
import numpy as np
from common import settings_file
from random import randint
import scipy.sparse as sparse
from tqdm import tqdm
from MemNN.model import SimpleQA_MemNN
from MemNN.input import symbols_bag, ngram_bag, g_q, f_y, negative_exemples_generation, g_q_single_question

logging.basicConfig(filename='debug.log',level=logging.DEBUG)
with open(settings_file()) as settings_f:
    settings = json.load(settings_f)

flags = tf.app.flags

tf.flags.DEFINE_float("lamb_da", 0.8, "hyperparameter lamb_da")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("epochs", 10, "Number of epochs to train for.")
tf.flags.DEFINE_integer("state", 0, "0, if training; 1 if testing; 2 if answering")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_integer("checkpoint_steps", 1, "number of epoch to store the checkpoint")
tf.flags.DEFINE_integer("embedding_size", 128, "hyperparameter embedding_size d")
tf.flags.DEFINE_boolean("from_checkpoint", False, "if we read the model from the checkpoint")
tf.flags.DEFINE_string("knowledgeBase", settings["kb"], "location of knowledge base")
tf.flags.DEFINE_string("train_set", settings['train'], "location of train dataset")
tf.flags.DEFINE_string("test_set", settings['test'], "location of test dataset")
tf.flags.DEFINE_string("summary", "./summary", "location of tensorboard summary")
tf.flags.DEFINE_string("checkpoint_folder", "./checkpoint", "checkpoint folder")
tf.flags.DEFINE_string("model_name", "QA", "model's name")


FLAGS = flags.FLAGS
initializer = tf.random_normal_initializer(mean=0, stddev=1/FLAGS.embedding_size)

############## generation of bag of word, bag of symbols #############


if FLAGS.state == 0: 

    # ---------------- training ----------------- #

    with open(settings["label2tid"], encoding="utf8") as f_in:
        film_labels = json.load(f_in)
    film_labels = list(film_labels.keys())

    # collect vocabulary and word2index
    (vocab, word2idx) = ngram_bag([FLAGS.train_set, FLAGS.test_set], film_labels)

    # collect symbol bag and symbol2index
    (symbols, symbol2index) = symbols_bag(FLAGS.knowledgeBase)

    # ---------------- save vocabulary, symbols bag and mapping functions (word -> index) and (symbol -> index) ----------------- #

    with open(settings['vocabulary'], "w", encoding="utf8") as f_out:
        for voc in vocab:
            print(voc, file=f_out, sep='\t')

    with open(settings['symbols'], "w", encoding="utf8") as f_out:
        for sym in symbols:
            print(sym, file=f_out, sep='\t')
    
    json.dump(word2idx, open(settings["word2index"], "w", encoding="utf8"))
    json.dump(symbol2index, open(settings["symbol2index"], "w", encoding="utf8"))

    # ---------------- generate mapping function (NGram <-> mid or imdbID) ----------------- #

    ngram2imdb = dict()
    ng2mid = dict()
    mid2ng = dict()
        
    mid2ngram = json.load(open(settings["mid2label"], encoding="utf8"))
    imdb2ngram = json.load(open(settings["imdbID2label"], encoding="utf8"))

    for sym in symbols:
        try:
            if isinstance(mid2ngram[sym], list):
                mid2ng[sym] = mid2ngram[sym]
                for alias in mid2ng[sym]:
                    ng2mid[alias] = sym
            else:
                mid2ng[sym] = [mid2ngram[sym]]
                ng2mid[mid2ngram[sym]] = sym
        except:
            pass

    for imdb in imdb2ngram.keys():
        ngram2imdb[imdb2ngram[imdb]] = imdb

    mid2ng.update(imdb2ngram)
    ng2mid.update(ngram2imdb)
    print ("ng2mid: ",len(ng2mid.keys()))
    print ("mid2ng: ", len(mid2ng.keys()))

    # ---------------- save mapping function (NGram <-> mid or imdbID) ----------------- #

    json.dump(mid2ng, open(settings["id2label"], "w", encoding="utf8"))
    json.dump(ng2mid, open(settings["label2id"], "w", encoding="utf8"))

else:
    # ---------------- load vocabulary, symbols bag and mapping functions (word -> index) and (symbol -> index) ----------------- #

    with open(settings['vocabulary'], encoding="utf8") as f_out:
        vocab = f_out.readlines()
        vocab = [i[:-1] for i in vocab]

    with open(settings['symbols'], encoding="utf8") as f_out:
        symbols = f_out.readlines()
        symbols = [i[:-1] for i in symbols]

    word2idx = json.load(open(settings["word2index"], encoding="utf8"))
    symbol2index = json.load(open(settings["symbol2index"], encoding="utf8"))

    # ---------------- load mapping function (NGram <-> mid or imdbID) ----------------- #

    mid2ng = json.load(open(settings["id2label"], encoding="utf8"))
    ng2mid = json.load(open(settings["label2id"], encoding="utf8"))
    print ("load ng2mid: ",len(ng2mid.keys()))
    print ("load mid2ng: ", len(mid2ng.keys()))

############## vectorization of knowledge base and dataset #############

# ------------- vectorize knowledge with symbol bag ------------ #

(kb_vec, kb_size, candidate_vec, responses) = f_y(symbol2index, FLAGS.knowledgeBase)
print ("knowledgeBase has been vectorized")
print ("knowledgeBase size: %d" % kb_size)

tf.set_random_seed(FLAGS.random_state)

if FLAGS.state == 0:

    batch_size = FLAGS.batch_size
    # ---------------- vectorize training dataset ----------------- #

    (train_vec_f, train_vec_q, train_size) = g_q(symbol2index, word2idx, FLAGS.train_set)
    (test_vec_f, test_vec_q, test_size) = g_q(symbol2index, word2idx, FLAGS.test_set)
    print ("train dataset has been vectorized")
    print ("train dataset size: %d" % train_size)

    # ---------------- generate negative examples ----------------- #

    (ng_vec, ng_size) = negative_exemples_generation(symbol2index, FLAGS.knowledgeBase)
    print ("negative exemples has been imported from knowledge base")

    # --------------------- generate batches ---------------------- #

    batches = zip(range(0, train_size-batch_size, batch_size), range(batch_size, train_size, batch_size))
    batches = [(start, end) for start, end in batches]

    test_batches = zip(range(0, test_size-batch_size, batch_size), range(batch_size, test_size, batch_size))
    test_batches = [(start, end) for start, end in test_batches]

random.seed()
vocabulary_size = len(vocab)
symbol_size = len(symbols)
checkpoint_steps = FLAGS.checkpoint_steps
checkpoint_dir = FLAGS.checkpoint_folder

################## MemNN model initialization ##################

parameters = dict()
parameters["symbol_size"] = symbol_size
parameters["initializer"] = initializer
parameters["vocabulary_size"] = vocabulary_size
parameters["kb_vec"] = kb_vec
parameters["kb_size"] = kb_size
parameters["candidate_vec"] = candidate_vec
parameters["responses"] = responses
parameters["word2index"] = word2idx
parameters["symbol2index"] = symbol2index
parameters["ngram2id"] = ng2mid

print ("------------------------")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.495)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = SimpleQA_MemNN(parameters, FLAGS, sess)

################## MemNN model train/test/answer ##################

    train_writer = tf.summary.FileWriter(FLAGS.summary + '/train', graph=sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summary + '/test', graph=sess.graph)
    saver = tf.train.Saver()

    # ------------- load trained model if FLAGS.from_checkpoint is True -------------- #

    if FLAGS.from_checkpoint:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print ("restoring checkpoint")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            pass

    print ("------------------------")

    ###################### MemNN train process ######################
    if FLAGS.state == 0:

        train_summary_index = 0
        test_summary_index = 0 
        for t in range(1, FLAGS.epochs+1):

            # ------------- shuffle batches -------------- #

            np.random.shuffle(batches)
            np.random.shuffle(test_batches)
            index = np.arange(np.shape(ng_vec)[0])
            np.random.shuffle(index)
            ng_vec = ng_vec[index, :]
            
            # --------------- train: feed data ---------------- #

            train_nombre = 0
            total_cost = 0.0
            for (start, end) in batches[:-1]:
                train_nombre += end-start
                f = train_vec_f[start:end].astype(np.float32)
                q = train_vec_q[start:end].astype(np.float32)
                n = ng_vec[start: end].astype(np.float32)
                cost_t, summary = model.batch_fit(f.todense(), n.todense(), q.todense())
                train_writer.add_summary(summary, train_summary_index)
                train_summary_index += 1
                if train_nombre % (batch_size*200) == 0:
                    print('Epoch:', t, " batch:", train_nombre/batch_size+1, "/", len(batches), ' cost_t:',cost_t, ' average_cost:', cost_t/batch_size)
                total_cost += cost_t 

            # --------------- valid: feed data ---------------- #

            valid_nombre = 0
            valid_cost = 0.0
            for (start, end) in test_batches[:100]:
                valid_nombre += end-start
                f = test_vec_f[start:end].astype(np.float32)
                q = test_vec_q[start:end].astype(np.float32)
                n = ng_vec[start: end].astype(np.float32)
                cost_t, summary = model.loss(f.todense(), n.todense(), q.todense())
                test_writer.add_summary(summary, test_summary_index)
                test_summary_index += 1
                valid_cost += cost_t 

            print('-----------------------')
            print('Epoch:', t)
            print('average cost of training set:', total_cost/train_nombre)
            print('average cost of valid set:', valid_cost/valid_nombre)
            print('-----------------------')

            # --------------- save model's checkpoint ---------------- #

            logging.info("Epoch: ", t, " loss = ", valid_cost/valid_nombre)
            if (t+1)%checkpoint_steps == 0:
                saver.save(sess, checkpoint_dir+'/model.ckpt', global_step=t+1)

    ###################### MemNN test process ######################
    elif FLAGS.state == 1:

        test = open(FLAGS.test_set, encoding="utf8").readlines()
        correct_number = 0
        sampled_test = random.sample(range(len(test)), 1000)
        for index in tqdm(sampled_test):
            line = test[index]
            splited_line = line.split("\t")
            result = splited_line[2]
            question = splited_line[3]
            if model.evaluate(result, question):
                correct_number += 1

        print('-----------------------')
        print('Test')
        print("Accuracy: ", correct_number/len(sampled_test))
        print('-----------------------')

    ###################### MemNN Answer process ######################
    elif FLAGS.state == 2:

        while 1:
            try:
                input_question = input("Enter your question: ")
                (max_similarity, best_candidate, best_fact) = model.find_most_probable_candidate(input_question)

                if max_similarity > 0:
                    print (responses[best_candidate])
                    for obj in responses[best_candidate]:
                        print (mid2ng[obj])
                    print ("cosine similarity: ", max_similarity)
                    print (best_fact)
                else:
                    print ("I don't know")
            except:
                print ("sorry, I can't reponse this question.")