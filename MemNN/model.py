import tensorflow as tf
from MemNN.output import g_q_single_question, candidate_generation
import random

def normalize_vector(vector_to_normalize):
    norm = tf.sqrt(tf.reduce_sum(tf.square(vector_to_normalize), 1, keep_dims=True))
    normalized_vector = vector_to_normalize/norm
    return normalized_vector

def cosine_similarity(labels, predictions):
    normalized_labels = normalize_vector(labels)
    normalized_predictions = normalize_vector(predictions)
    similarity = tf.matmul(normalized_labels, tf.transpose(normalized_predictions))
    return similarity

class SimpleQA_MemNN(object):
    def __init__(self, parameters, config, session):
        """Creates an embedding based Memory Network
        Args:
            parameters:
                vocabulary_size : size of bag of word
                symbol_size: size of bag of symbol
                initializer: random initializer for embedding matrix
                kb_vec: knowledge base vector
                kb_size: size of knowledge Base
                candidate_vec: candidate(subject, relationship) vector
                word2index: map word to index
                symbol2index: map mid/IMDBid to index
                ngram2id: map labels to mid or IMDBid
                responses: map numero of question to response

            config:
                embedding_size: dimension of embedding matrix
                lamb_da: margin in ranking loss function
                model_name

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.
        """

        self._embedding_size = config.embedding_size
        self._lambda = config.lamb_da
        self._name = config.model_name
        self._vocabulary_size = parameters["vocabulary_size"]
        self._init = parameters["initializer"]
        self._symbole_size = parameters["symbol_size"]

        self._kb_vec = parameters["kb_vec"]
        self._kb_size = parameters["kb_size"]
        self._candidate_vec = parameters["candidate_vec"]
        self._word2index = parameters["word2index"]
        self._symbol2index = parameters["symbol2index"]
        self._ngram2id = parameters["ngram2id"]
        self._responses = parameters["responses"]

        # build placeholder and initializer embedding matrices
        self._build_inputs()
        self._build_vars()

        # optimizer
        self._opt = tf.train.AdadeltaOptimizer()

        # cosine similarity
        cosine_similarity_positive =  self.similarity_calcul(self._facts, self._questions)
        cosine_similarity_negative =  self.similarity_calcul(self._negative_facts, self._questions)
        loss = tf.nn.relu(tf.add(tf.subtract(self._lambda, cosine_similarity_positive), cosine_similarity_negative))
        loss_sum = tf.reduce_sum(loss, name="loss_sum")
        loss_unity = tf.reduce_mean(loss)
        tf.summary.scalar('average loss', loss_unity)

        # loss op
        loss_op = loss_sum
        train_op = self._opt.minimize(loss_op)

        # assign ops
        self.loss_op = loss_op
        self.similarity_op = cosine_similarity_positive
        self.train_op = train_op
        self.merged_summary = tf.summary.merge_all()

        # variable intialization
        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)


    def _build_inputs(self):
        # build placeholder for input vectors
        self._questions = tf.placeholder(tf.float32, [None, self._vocabulary_size], name="queries")
        self._facts = tf.placeholder(tf.float32, [None, self._symbole_size], name="facts")
        self._negative_facts = tf.placeholder(tf.float32, [None, self._symbole_size], name="negative_facts")

        self._wiki_questions = tf.placeholder(tf.float32, [None, self._vocabulary_size], name="wiki_queries")
        self._wiki_similar_questions = tf.placeholder(tf.float32, [None, self._vocabulary_size], name="wiki_similar_queries")
        self._wiki_unsimilar_questions = tf.placeholder(tf.float32, [None, self._vocabulary_size], name="wiki_unsimilar_queries")

    def _build_vars(self):

        # build embedding matrix
        with tf.variable_scope(self._name):
            self.Wv = tf.Variable(self._init([self._embedding_size, self._vocabulary_size]), name="Wv")
            self.Ws = tf.Variable(self._init([self._embedding_size, self._symbole_size]), name="Ws")
            self.Wvs = tf.concat([self.Wv, self.Ws], 1)

    def similarity_calcul(self, facts, questions):

        with tf.variable_scope(self._name):
            labels = tf.matmul(self.Wv,tf.transpose(questions))
            predictions = tf.matmul(self.Ws,tf.transpose(facts))
            cosine_sim =  cosine_similarity(tf.transpose(labels), tf.transpose(predictions))
            return tf.diag_part(cosine_sim)

    def vairable_summaries(self, var):

        with tf.variable_scope(self._name):

            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def batch_fit(self, facts, negative_facts, questions):

        # train with ranking loss function
        feed_dict = {self._questions: questions, self._facts: facts, self._negative_facts: negative_facts}
        loss, _, summary = self._sess.run([self.loss_op, self.train_op, self.merged_summary], feed_dict=feed_dict)
        return loss, summary
    
    def loss(self, facts, negative_facts, questions):

        # calculate loss
        feed_dict = {self._questions: questions, self._facts: facts, self._negative_facts: negative_facts}
        loss, summary= self._sess.run([self.loss_op, self.merged_summary], feed_dict=feed_dict)
        return loss, summary

    def similarity(self, facts, questions):
        
        # calcul similarity for answering questions
        feed_dict = {self._questions: questions, self._facts: facts}
        similar= self._sess.run([self.similarity_op], feed_dict=feed_dict)
        return similar

    def find_most_probable_candidate(self, question):

        question_vectorized = g_q_single_question(self._word2index, question)
        candidates_generated = candidate_generation(self._candidate_vec, self._kb_size, len(self._symbol2index), self._ngram2id , self._symbol2index, question)
        max_similarity = 0

        for candidate in candidates_generated:
            candidate_fact = self._kb_vec[candidate]
            similarity_rate = self.similarity(candidate_fact.todense(), question_vectorized.todense())
            if max_similarity < similarity_rate[0][0]:
                max_similarity = similarity_rate[0][0]
                best_candidate = candidate
                best_fact = candidate_fact
        
        return (max_similarity, best_candidate, best_fact)
    
    def evaluate(self, obj, question):
        try:
            (max_similarity, best_candidate, best_fact) = self.find_most_probable_candidate(question)
            if max_similarity > 0 and obj in self._responses[best_candidate]:
                return True
            return False
        except:
            return False