from pyspark import SparkContext

class TFIDFProcessor:

@staticmethod
def compute_tf_idf(sc, file_path):
    """
    
    """
    rdd = sc.textFile(file_path)
    frequency_by_document = rdd.flatMap(lambda line: [[word + "_" + line.split(' ')[0], 1] for word in line.split(' ')[1:] if word != ''])\
    .reduceByKey(lambda x, y: x + y)\
    .map(lambda x: [x[0].split("_")[1], [x[0].split("_")[0],  x[1]]])
    term_frequency = rdd.map(lambda line: [line.split(' ')[0], len(line.split(' ')[1:])])\
                    .join(frequency_by_document)\
                    .map(lambda pair: [pair[1][1][0], [pair[0], pair[1][1][1], pair[1][0]]])
    total_documents = rdd.count()
    word_counts = rdd \
    .flatMap(lambda line: [[term, 1] for term in line.split(' ')[1:] if term != ""]) \
    .reduceByKey(add) \
    .map(lambda x: [x[0], log(total_documents / x[1])])
    tf_idf_sparse_matrix = term_frequency.join(word_counts)\
    .map(transform_tf_idf)\
    .collect()
    all_terms  = rdd.flatMap(extract_terms).filter(filter_empty_strings).distinct()
    all_doc_ids = rdd.map(lambda line: line.split(' ')[0])
    terms_map = all_terms.zipWithIndex().collectAsMap()
    doc_map = all_doc_ids.zipWithIndex().collectAsMap()
    return make_matrix(tf_idf_sparse_matrix, terms_map, doc_map))    


def extract_terms(line):
    terms = line.split(' ')[1:]
    return terms

def filter_empty_strings(term):
    return term != ''

def extract_terms_with_doc_id(line):
    data = line.split(' ')
    doc_id, terms = data[0], data[1:]
    return [term + ",," + doc_id for term in terms if term != '']

def split_term_doc_id_pair(pair):
    term, doc_id = pair.split(",,")
    return (term, [doc_id])

def transform_tf_idf(joined_pair):
    term = joined_pair[0]
    idf = joined_pair[1][1]
    tf_doc = joined_pair[1][0][1]
    doc_total_terms = joined_pair[1][0][2]
    document_id = joined_pair[1][0][0]
    return [term, document_id , idf * (tf_doc / doc_total_terms)]

def make_matrix(data, term_map, document_map):
    matrix = np.zeros(shape=(len(document_map), len(term_map)))
    for (term, doc_id, value) in data:
            doc_index = document_map[doc_id]
            term_index = term_map[term]
            matrix[doc_index][term_index] = value
    return matrix