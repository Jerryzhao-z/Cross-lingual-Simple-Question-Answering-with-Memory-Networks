# import for python 2.7
#from io import open
#
import itertools
import scipy.sparse as sparse
from MemNN.common import split_line, clean_words

def process_fact(line):
    # Arg: "entity \t rel \t obj1 obj2 ob3 ..."
    # Return: (entity rel obj1 obj2 obj3 ....)
    [entity, rel, obj] = line.rstrip().split('\t')
    return entity, rel, obj.split(' ')


def symbols_bag(kb):
    """

    Note:
        collect symbols from knowledge base

    Args:
        kb : knowledge base's path

    Returns:
        symbol_list, symbol2index

    """
    print ("processing Knowledge base to bag-of-symbole")

    # symbol_list
    all_symbol = set()
    with open(kb, encoding="utf8") as f_in:
        for l in f_in:
            entity, rel, objs = process_fact(l)
            all_symbol.update([entity, rel])
            all_symbol.update(objs)
    symbol_list = list(all_symbol)
    print ("%d symbols have been processed" % len(symbol_list))
    symbol_list.sort()

    # symbol2index
    symbol2index = {}
    symbol2index.update(zip(symbol_list, itertools.count()))
    return symbol_list, symbol2index


def ngram_bag(corpus_list, labels):
    """

    Note:
        collect ngrams from dataset

    Args:
        corpus_list: list of dataset similar to SimpleQuestion train/test/valid
        labels: labels of entity in Knowledge base

    Returns:
        vocabulary, voc2index

    """
    # vocabulary
    print ("processing corpus to bag-of-words")
    words = set()
    line_ctr = itertools.count()

    # words in questions
    for ds in corpus_list:
        with open(ds, encoding="utf8") as in_f:
            for line in in_f:
                try:
                    line_number = next(line_ctr)
                    words.update(clean_words(split_line(line)))
                except IndexError:
                    print ("Index Error in line %d" % line_number)

    # word in labels
    for l in labels:
        words.update(l.split())
    vocabulary = list(words)
    print ("%d words have been processed" % len(vocabulary))
    vocabulary.sort()

    # voc2index
    voc2index = {}
    voc2index.update(zip(vocabulary, itertools.count()))
    return vocabulary, voc2index

# preprocessing Freebase facts: transform a fact (s, r, {o1, ... ok} to vector with a bag-of-symbole
def f_y(symbols2index, kb):
    """

    Note:
        preprocessing knowledge base

    Args:
        symbols2index: mapping object
        kb: knowledge base's path

    Returns:
        mx: knowledge matrix
        knowledgebase_size: number of facts 
        candidate_mx: subject and relationship matrix
        responses: mapping number of fact to objects

    """
    line_ctr = itertools.count()
    data_tuples = list()
    responses = dict()
    candidate_tuple = list()

    with open(kb, encoding="utf8") as f_in:
        for l in f_in:
            entity, rel, objs = process_fact(l)
            l = next(line_ctr)
            data_tuples.append((1.0, l, symbols2index[entity]))
            data_tuples.append((1.0, l, symbols2index[rel]))
            candidate_tuple.append((1.0, l, symbols2index[entity]))
            candidate_tuple.append((1.0, l, symbols2index[rel]))
            data_tuples.extend([(1./len(objs), l, symbols2index[o]) for o in objs])
            responses[l] = objs
    data, row, col = zip(*data_tuples)
    candidate_data, candidate_row, candidate_col = zip(*candidate_tuple)

    knowledgebase_size = next(line_ctr)
    symbol_size = len(symbols2index.keys())

    mx = sparse.csr_matrix((data, (row, col)), shape=(knowledgebase_size,symbol_size))
    candidate_mx = sparse.csr_matrix((candidate_data, (candidate_row, candidate_col)), shape=(knowledgebase_size,symbol_size))
    return mx, knowledgebase_size, candidate_mx, responses

def f_y_facts(symbols2index, dataset):
    """

    Note:
        preprocessing facts in dataset

    Args:
        symbols2index: mapping object
        dataset: dataset similar to SimpleQuestion train/valid/test

    Returns:
        mx: fact matrice

    """
    line_ctr = itertools.count()
    data_tuples = list()
    for l in dataset:
        entity, rel, obj, question = l.rstrip().split('\t')
        l = next(line_ctr)
        data_tuples.append((1.0, l, symbols2index[entity]))
        data_tuples.append((1.0, l, symbols2index[rel]))
        data_tuples.append((1.0, l, symbols2index[obj]))

    data, row, col = zip(*data_tuples)
    mx = sparse.csr_matrix((data, (row, col)))
    return mx

def g_q(symbols2index, voc2index, dataset):
    """

    Note:
        preprocessing dataset

    Args:
        symbols2index: map symbol to index
        voc2index: map word to index
        dataset: dataset similar to SimpleQuestion train/valid/test

    Returns:
        f_mx: fact matrice
        q_mx: question matrice
        M: number of records in dataset

    """
    line_ctr = itertools.count()
    data_tuples = list()
    fact_tuples = list()
    with open(dataset, encoding="utf8") as in_f:
        for line in in_f:
            l = next(line_ctr)
            fact_tuples.extend([(1, l, symbols2index[s]) for s in line.split("\t")[0:3]])
            data_tuples.extend([(1, l, voc2index[w]) for w in clean_words(split_line(line))])

    f_data, f_row, f_col = zip(*fact_tuples)
    q_data, q_row, q_col = zip(*data_tuples)
    M = next(line_ctr)
    N = len(symbols2index.keys())
    O = len(voc2index.keys())

    f_mx = sparse.csr_matrix((f_data, (f_row, f_col)), shape=(M, N))
    q_mx = sparse.csr_matrix((q_data, (q_row, q_col)), shape=(M, O))
    return f_mx, q_mx, M

def g_q_single_question(voc2index, question):
    """

    Note:
        preprocessing single question

    Args:
        voc2index: map word to index
        question: question in natural language

    Returns:
        q_mx: question vector

    """
    data_tuples = list()

    data_tuples.extend([(1, 0, voc2index[w]) for w in clean_words(question.strip().lower().split(' ')) if voc2index[w]])
    q_data, q_row, q_col = zip(*data_tuples)

    O = len(voc2index.keys())

    q_mx = sparse.csr_matrix((q_data, (q_row, q_col)), shape=(1, O))
    return q_mx

def negative_exemples_generation(symbols2index, kb):
    """

    Note:
        generate negative examples from knowledge base

    Args:
        symbols2index: map symbol to index
        kb: knowledge base's path

    Returns:
        mx: negative example matrice
        M: number of negative examples

    """
    line_ctr = itertools.count()
    data_tuples = list()
    with open(kb, encoding="utf8") as f_in:
        for l in f_in:
            entity, rel, objs = process_fact(l)
            for o in objs:
                l = next(line_ctr)
                data_tuples.append((1.0, l, symbols2index[entity]))
                data_tuples.append((1.0, l, symbols2index[rel]))
                data_tuples.append((1.0, l, symbols2index[o]))

    data, row, col = zip(*data_tuples)
    M = next(line_ctr)
    N = len(symbols2index.keys())
    mx = sparse.csr_matrix((data, (row, col)), shape=(M,N))
    return mx, M

