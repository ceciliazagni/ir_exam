"""
A Boolean Information Retrieval System with the possibility to add documents
"""

from functools import total_ordering, reduce
import bisect
import csv
import re
import time
import logging
import pickle
from datetime import datetime
import sys
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Configure the logging 
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f'document_processing_{current_datetime}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



"""Postings"""

# This is a class representing an element of a posting list

@total_ordering
class Posting:

    def __init__(self, docID, positions):
        self._docID = docID
        self._positions = positions # List of the positions of the associated Term in the document DocID

    # Method that adds a position to the list of positions, maintaining the order
    def add_position(self, position):
        bisect.insort(self._positions, position) # Insert in order

    # Method that returns the list of positions
    def get_positions(self):
        return self._positions

    # Method that retrieves the corresponding title from the corpus based on the document ID
    def get_from_corpus(self, corpus):
        return corpus[self._docID]

    def __eq__(self, other):
        return self._docID == other._docID

    def __gt__(self, other):
        return self._docID > other._docID

    def __repr__(self):
        return f"{self._docID}:{self._positions}"



"""### Posting Lists"""

# This is a class representing a posting list

class PostingList:

    def __init__(self):
        self._postings = [] # A posting list is a list of postings

    @classmethod
    def from_posting_list(cls, postingList):
        plist = cls()
        postingList.sort() # We want the posting list to be ordered
        plist._postings = postingList
        return plist

    # Method that removes a posting with a specific document ID (key) from the posting list
    def remove_posting(self, key):
        for posting in self._postings:
            if posting._docID == key:
                self._postings.remove(posting)
                return
        raise KeyError(f"Document ID {key} not found in the posting list")

    # Metod that merges two posting lists into one, maintaining the order of document IDs
    def merge(self, other):
        i = 0
        j = 0
        merged = []
        while i < len(self._postings) and j < len(other._postings):
            if self._postings[i] == other._postings[j]:
                merged.append(self._postings[i])
                i += 1
                j += 1
            elif self._postings[i] < other._postings[j]:
                merged.append(self._postings[i])
                i += 1
            else:
                merged.append(other._postings[j])
                j += 1
        while i < len(self._postings):
            merged.append(self._postings[i])
            i += 1
        while j < len(other._postings):
            merged.append(other._postings[j])
            j += 1
        self._postings = merged

    # Method that computes the intersection of two posting lists
    def intersection(self, other):
        intersection = []
        i = 0
        j = 0
        while (i < len(self._postings) and j < len(other._postings)):
            if (self._postings[i] == other._postings[j]):
                intersection.append(self._postings[i])
                i += 1
                j += 1
            elif (self._postings[i] < other._postings[j]):
                i += 1
            else:
                j += 1
        return PostingList.from_posting_list(intersection)

    # Method that computes the union of two posting lists
    def union(self, other):
        union = []
        i = 0
        j = 0
        while (i < len(self._postings) and j < len(other._postings)):
            if (self._postings[i] == other._postings[j]):
                union.append(self._postings[i])
                i += 1
                j += 1
            elif (self._postings[i] < other._postings[j]):
                union.append(self._postings[i])
                i += 1
            else:
                union.append(other._postings[j])
                j += 1
        for k in range(i, len(self._postings)):
            union.append(self._postings[k])
        for k in range(j, len(other._postings)):
            union.append(other._postings[k])
        return PostingList.from_posting_list(union)

    # Method that retrieves the corresponding titles from the corpus for all document IDs in the posting list
    def get_from_corpus(self, corpus):
        return list(map(lambda x: x.get_from_corpus(corpus), self._postings))
    
    # Method that returns the posting corresponding to the given docID
    def __getitem__(self, key):
        for posting in self._postings:
            if posting._docID == key:
                return posting
        return None

    def __repr__(self):
        return ", ".join(map(str, self._postings))


# Returns the posting list of the terms that are consecutive in pl1 and pl2
def consecutive(pl1, pl2):
        consecutive = []
        i = 0
        j = 0
        while (i < len(pl1._postings) and j < len(pl2._postings)):
            if (pl1._postings[i] == pl2._postings[j]):
                k = 0
                l = 0
                while (k < len(pl1._postings[i]._positions) and l < len(pl2._postings[j]._positions)):
                    if (pl1._postings[i]._positions[k]+1 == pl2._postings[j]._positions[l]):
                        consecutive.append(pl2._postings[j])
                        break
                    elif (pl1._postings[i]._positions[k]+1 < pl2._postings[j]._positions[l]):
                        k += 1
                    else:
                        l += 1
                i += 1
                j += 1
            elif (pl1._postings[i] < pl2._postings[j]):
                i += 1
            else:
                j += 1
        return PostingList.from_posting_list(consecutive) if consecutive else None



"""Terms"""

# This is a class representing a Term. A Term is a word with the associated relative posting list

class ImpossibleMergeError(Exception):
    pass

@total_ordering
class Term:

    def __init__(self, term, posting_list):
        self._term = term # The term (word) itself
        self._posting_list = posting_list # The PostingList associated with the term

    # Method that merges the PostingList of the current term with the PostingList of another term.
    def merge(self, other):
        if (self._term == other._term):
            self._posting_list.merge(other._posting_list)
        else:
            raise ImpossibleMergeError

    def __eq__(self, other):
        return self._term == other._term

    def __gt__(self, other):
        return self._term > other._term

    def __repr__(self):
        return self._term + ": " + repr(self._posting_list)



"""B-Tree structure"""

# A class defining a B-Tree node

class BTreeNode:
    def __init__(self, T_MAX, leaf=True):
        self.t = T_MAX # maximum degree of the B-Tree the node is part of
        self.keys = [] # list of the keys in the node
        self.children = [] # list of the children of the node
        self.leaf = leaf # set to true if the node if a leaf of the B-Tree

    # Method used to insert a key in the B-tree
    # It traverses the tree recursively, splitting nodes when necessary to maintain the B-tree properties
    def insert(self, key):
        if not self.leaf:
            i = len(self.keys) - 1
            while i >= 0 and key < self.keys[i]:
                i -= 1
            i += 1
            if len(self.children[i].keys) == 2 * self.t - 1:
                self.split_child(i)
                if key > self.keys[i]:
                    i += 1
            self.children[i].insert(key)
        else:
            i = len(self.keys) - 1
            while i >= 0 and key < self.keys[i]:
                i -= 1
            self.keys.insert(i + 1, key)
    
    # Method used when a child node is full
    # It splits the child into two nodes and redistributes the keys appropriately
    def split_child(self, i):
        y = self.children[i]
        z = BTreeNode(self.t,leaf=y.leaf)
        self.children.insert(i + 1, z)
        self.keys.insert(i, y.keys[self.t - 1])
        z.keys = y.keys[self.t:]
        y.keys = y.keys[:self.t - 1]
        if not y.leaf:
            z.children = y.children[self.t:]
            y.children = y.children[:self.t]
    
    # Method that searches for a key in the B-tree and returns the node containing the key and the index of the key within the node
    def search_key(self, key):
        i = 0
        while i < len(self.keys) and key > self.keys[i]:
            i += 1
        if i < len(self.keys) and key == self.keys[i]:
            return self, i
        elif self.leaf:
            return None, -1
        else:
            return self.children[i].search_key(key)

    # Method that recursively counts the number of nodes in the subtree rooted at the current node
    def count_nodes(self):
        count = 1
        if not self.leaf:
            for child in self.children:
                count += child.count_nodes()
        return count


# Class that represents the B-tree itself
class BTree:
    def __init__(self, T_MAX):
        self.t = T_MAX # Maximum degree of the B-tree
        self.root = BTreeNode(self.t,leaf=True) # Root node

    # Method that inserts a key into the B-tree
    # If the root is full, it creates a new root and splits the old root
    def insert(self, key):
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(self.t,leaf=False)
            new_root.children.append(self.root)
            new_root.split_child(0)
            self.root = new_root
        self.root.insert(key)

    def search_key(self, key):
        return self.root.search_key(key)

    def count_nodes(self):
        return self.root.count_nodes()
    
    # Method that returns the total number of keys in the B-tree by calling a recursive helper method
    def count_keys(self):
        return self._count_keys_recursive(self.root)

    def _count_keys_recursive(self, node):
        if node is None:
            return 0
        count = len(node.keys)
        for child in node.children:
            count += self._count_keys_recursive(child)
        return count

"""
# Function that merges the keys from treeB into treeA
def merge_trees(treeA, treeB):
    def merge_node(treeA, nodeB):
        for key in nodeB.keys:
            treeA.insert(key)
        if not nodeB.leaf:
            for childB in nodeB.children:
                merge_node(treeA, childB)
    merge_node(treeA, treeB.root)
    return treeA
 """
 
    
"""
# Function that finds the maximum key in a B-tree, given its root
# It traverses the rigthmost path of the tree and and returs the last key when it reaches the leaf node
def find_max_key(node):
    if node is None:
        return None
    while not node.leaf:
        node = node.children[-1]
    return node.keys[-1]
"""


"""Inverted Index"""
"""
def normalize(text):
    no_punctuation = re.sub(r'[^\w^\s^-]','',text) # remove all characters that are not alphanumeric, whitespace, or hyphen
    downcase = no_punctuation.lower() # convert the remaining text to lowercase
    return downcase


def tokenize(movie):
    text = normalize(movie.description)
    return list(text.split())
"""

def normalize_stem_tokenize(text):
    no_punctuation = re.sub(r'[^\w^\s^-]', '', text)
    downcase = no_punctuation.lower()
    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()
    # Apply stemming to each word
    stemmed_words = [stemmer.stem(word) for word in list(downcase.split())]
    return stemmed_words

def document_normalize_stem_tokenize(document):
    return normalize_stem_tokenize(document.description)

# Class that levereges the B-Tree data structure in order to build an inverted index

class InvertedIndex:

    def __init__(self, dict_order=256):
        self._dictionary = BTree(dict_order) # the dictionary is a B-Tree of maximum deree dict_order
        self._dict_order = dict_order # Maximum degree of the dictionary's B-Tree

    # Method used to create an inverted index from a given corpus
    # It takes the corpus and an optional dict_order parameter
    # It iterates through the corpus, tokenizes the documents, and creates posting lists for each term
    # These posting lists are then inserted as keys in a B-tree in order to create the inverted index
    @classmethod
    def from_corpus(cls, corpus, dict_order=256):
        idx = cls()
        idx._dict_order = dict_order
        idx._dictionary = BTree(dict_order)
        start_time = time.perf_counter()
        for docID, document in enumerate(corpus):
            tokens = document_normalize_stem_tokenize(document)
            term_positions = {}
            for i, token in enumerate(tokens):
                if token in term_positions:
                    term_positions[token].append(i)
                else:
                    term_positions[token] = [i]
            for term, positions in term_positions.items():
                t = Term(term, PostingList.from_posting_list([Posting(docID, positions)]))
                node, pos = idx._dictionary.search_key(t)
                if node is not None:
                    node.keys[pos].merge(t)
                else:
                    idx._dictionary.insert(t)
            if (docID % 1000 == 0):
                mid_time = time.perf_counter()
                elapsed_time = mid_time - start_time
                logging.info(f'After {elapsed_time}s, {docID} inserted')
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logging.info(f'Corpus indexed in {elapsed_time}s')
        return idx

    # Method that takes a node of an other inverted index and merges it and its children into self
    # It is usefull to merge two dictionaries when the given node is the root of another inverted index dictionary
    def merge_indexes(self,node):
        if node is not None:
            for term in node.keys:
                selfnode, pos = self._dictionary.search_key(term)
                if selfnode is not None:
                    selfnode.keys[pos].merge(term)
                else:
                    self._dictionary.insert(term)
            if not node.leaf:
                for child in node.children:
                    self.merge_indexes(child)

    # Method used to remove the specified list of document IDs from the posting lists of terms in the inverted index
    # It helps to handle document deletions efficiently
    def delete_docIDs_from_dictionary(self,docIDs_list):
        def del_docIDs_traverse_tree(node,docIDs_list):
            if node is not None:
                for term in node.keys:
                    term._posting_list._postings = [posting for posting in term._posting_list._postings if posting._docID not in docIDs_list]
                if not node.leaf:
                    for child in node.children:
                        del_docIDs_traverse_tree(child,docIDs_list)
        return del_docIDs_traverse_tree(self._dictionary.root,docIDs_list)

    # Method that traverses the inverted index to find the maximum document ID present in the posting lists
    # This is useful in order to build a dynamic index when I don't know the maximun doc ID in the curren inverted index
    def find_maximum_docID(self):
        max_docID = -1
        def find_max_traverse_tree(node):
            nonlocal max_docID   # nonlocal allows the find_max_traverse_tree function to modify the max_docID variable from the outer scope
            if node is not None:
                for term in node.keys:
                    for posting in term._posting_list._postings:
                        if posting._docID > max_docID:
                            max_docID = posting._docID
                if not node.leaf:
                    for child in node.children:
                        find_max_traverse_tree(child)

        find_max_traverse_tree(self._dictionary.root)
        return max_docID

    # Method that allows to retrieve the posting list associated with a given term key
    def __getitem__(self, key):
        node, pos = self._dictionary.search_key(key)
        if node is not None:
            return node.keys[pos]._posting_list
        else:
            return None

    def __repr__(self):
        return "A dictionary with " + str(self._dictionary.count_keys()) + " terms, in a Btree of order " + str(self._dict_order) + " and " + str(self._dictionary.count_nodes()) + " nodes"



"""### Dynamic Index"""
# This class contains two inverted indexes, a primary one and a auxiliary one, used to allow documents addiction
# The class is responsible for managing the two indexes and to make them "look like one single index" to the IR system

class DynamicIndex:

    def __init__(self, primary_index, num_documents, aux_index_order = 10):
        self._primary_index = primary_index # Primary inverted index, which contains the originally indexed data
        self._aux_index_order = aux_index_order # order of the auxiliary inverted index, which can be used for faster updates
        self._auxiliary_index = InvertedIndex(self._aux_index_order) # the auxiliary inverted index
        self._invalidation_vector = [False] * num_documents # A boolean vector that marks the deleted documents

    # Method that adds a new document to the dynamic index
    # It updates the auxiliary index with the new document's terms and appends False to the _invalidation_vector, indicating that the document is valid and not deleted
    def add_document(self, document):
        # Update the auxiliary index with the new document
        tokens = document_normalize_stem_tokenize(document)
        term_positions = {}
        for i, token in enumerate(tokens):
            if token in term_positions:
                term_positions[token].append(i)
            else:
                term_positions[token] = [i]
        for term, positions in term_positions.items():
            t = Term(term, PostingList.from_posting_list([Posting(len(self._invalidation_vector), positions)]))
            node, pos = self._auxiliary_index._dictionary.search_key(t)
            if node is not None:
                node.keys[pos].merge(t)
            else:
                self._auxiliary_index._dictionary.insert(t)
        # Update self._invalidation_vector
        self._invalidation_vector.append(False)

    def delete_document(self, doc_id):
        # Mark the corresponding bit in the invalidation vector as True
        self._invalidation_vector[doc_id] = True

    # Method that combines the data from the auxiliary index into the primary index while considering the invalidated documents
    def merge_aux_into_primary(self):
        # delete the invalidated dicIDs from the indexes
        deleted_docIDs = [index for index, value in enumerate(self._invalidation_vector) if value]
        self._primary_index.delete_docIDs_from_dictionary(deleted_docIDs)
        self._auxiliary_index.delete_docIDs_from_dictionary(deleted_docIDs)
        # merge auxiliary index into primary index
        self._primary_index.merge_indexes(self._auxiliary_index._dictionary.root)

    # Method that allows to retrieve a posting list for a given term from the dynamic index
    # It retrieves the posting lists from both the primary and auxiliary indexes, then applies the invalidation logic using the _invalidation_vector to remove deleted documents' data
    # It returns the combined posting list
    def __getitem__(self, key):
        primary_postings = self._primary_index[key]
        auxiliary_postings = self._auxiliary_index[key]
        union_postings = None
        if primary_postings and auxiliary_postings:
            union_postings = primary_postings.union(auxiliary_postings)
        elif primary_postings:
            union_postings = primary_postings
        elif auxiliary_postings:
            union_postings = auxiliary_postings
        if union_postings:
            for doc_id, invalidated in enumerate(self._invalidation_vector):
                if invalidated and union_postings[doc_id]:
                    union_postings.remove_posting(doc_id)
        return union_postings


    def __repr__(self):
        return f"Primary index: {self._primary_index}\nAuxiliary index: {self._auxiliary_index}\nInvalidation vector: {self._invalidation_vector.count(True)} invalidated elements."



"""Query parsing"""

# Function that takes a query as input and returns a list of tokens that represent the query elements
# The tokens include terms, Boolean operators, and parentheses
def parse_query(query):
    pattern = r'\w+|AND|OR|NOT|\(|\)'
    tokens = re.findall(pattern, query)
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
    output = []
    operators = []
    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()
    for token in tokens:
        if token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()
        elif token in precedence:
            while (operators and operators[-1] != '(' and
                   precedence[token] <= precedence.get(operators[-1], 0)):
                output.append(operators.pop())
            operators.append(token)
        else:
            output.append(stemmer.stem(token))
    while operators:
        output.append(operators.pop())
    return output


# Function that takes the output of the parse_query function and evaluates the Boolean expression defined by those tokens
# It returns a dictionary containing intermediate results
def evaluate_expression(expression):
    stack = []
    results = {}
    r=0
    for token in expression:
        if token in {'AND', 'OR'}:
            op2 = stack.pop()
            op1 = stack.pop()
            r += 1
            tmp = f'res{r}'
            stack.append(tmp)
            results[tmp] = (token,[op1,op2])
        elif token == 'NOT':
            op = stack.pop()
            r += 1
            tmp = f'res{r}'
            stack.append(tmp)
            results[tmp] = (token,op)
        else:
            stack.append(token)
    return results



"""### Reading the Corpus"""

class MovieDescription:

    def __init__(self, title, description):
        self.title = title
        self.description = description

    def __repr__(self):
        return self.title

def read_movie_descriptions():
    filename = 'plot_summaries.txt'         # id + description
    movie_names_file = 'movie.metadata.tsv' # id + data + description + ...

    # creo indice che collega id a titolo del film
    with open(movie_names_file, 'r', encoding="utf8") as csv_file:
        movie_names = csv.reader(csv_file, delimiter='\t')
        names_table = {}
        for name in movie_names:
            names_table[name[0]] = name[2]

    # creo la lista corpus, ogni suo elemento è di tipo MovieDescription, creato grazie all'indice precedente
    with open(filename, 'r', encoding="utf8") as csv_file:
        descriptions = csv.reader(csv_file, delimiter='\t')
        corpus = []
        for desc in descriptions:
            try:
                movie = MovieDescription(names_table[desc[0]], desc[1])
                corpus.append(movie)
            except KeyError:
                pass
        return corpus



"""Information Retrival system"""
# Class representing an Information Retrieval system that provides functionalities to manage and search through a corpus of documents using a dynamic index
class IRsystem:

    def __init__(self, corpus, dynamic_index):
        self._corpus = corpus # list of documents that the IR system will operate on
        self._index = dynamic_index # dynamic index used for document retrieval
    
    # Method that creates an instance of the IRsystem class using a provided corpus of documents
    # It constructs a dynamic index based on the given corpus and returns an initialized IRsystem object
    # The main_index_order and aux_index_order parameters determine the order of the main and auxiliary indexes used in the dynamic index
    @classmethod
    def from_corpus(cls, corpus, main_index_order = 256, aux_index_order = 10):
        index =  DynamicIndex(InvertedIndex.from_corpus(corpus,main_index_order),len(corpus),aux_index_order)
        return cls(corpus, index)

    # Method that searches for a document in the corpus based on its title and returns the document and its index if found
    def find_document_by_title(self, title):
        for idx, document in enumerate(self._corpus):
            if document.title == title:
                return document, idx
        return None, -1  # Return None if the movie title is not found

    # Method that adds a new document to the IR system
    # It updates the dynamic index with the new document and appends the document to the corpus
    def add_document(self, document):
        if document in self._corpus:
           print(f'Document {document} already present.')
        else:
          # Update the index with the new document
           self._index.add_document(document)
           # Update the corpus
           self._corpus.append(document)

    # Method tha deletes a document from the IR system based on its title
    def delete_document(self, title):
        document, doc_id = self.find_document_by_title(title)
        if document:
            self._index.delete_document(doc_id)
        else:
            print(f'Document titeld {title} not present in the corpus.')

    # Method that merges the auxiliary indexe in the primary index, without touching the corpus and the invalidation vector
    # This type of merge is faster, because there is no need to reindexing the documents in the corpus, but can't be definitive in order to avoid the fact that the corpus contains a too big amount of deleted documents
    def fast_merge_dynamic_index(self):
        # merge indexes
        self._index.merge_aux_into_primary()

    # Method that permanently deletes from the corpus the deleted documents, then completeley regenerates the dynamic index from the new corpus
    # This is not a proper merge, but a rebuilding of the index, so it is very slow, but it is necessary when a reindexing of the documents in the corpus is needed
    def complete_merge(self):
        # delete invalidated documents from corpus
        tmp = [docID for docID, validate in zip(self._corpus, self._index._invalidation_vector) if not validate]
        self._corpus = tmp
        # Rebuild the index
        main_index_order = self._index._primary_index._dict_order
        aux_index_order = self._index._aux_index_order
        self._index = DynamicIndex(InvertedIndex.from_corpus(self._corpus,main_index_order),len(self._corpus),aux_index_order)

    # Method that processes and answers a given query using the IR system's dynamic index
    # It supports both boolean and phrase queries, utilizing the handle_boolean_query and handle_phrase_query methods
    def answer_query(self, query):
        if 'AND' in query or 'OR' in query or 'NOT' in query and '"' not in query:
            # Handle boolean query
            boolean_query_result = self.handle_boolean_query(query)
            if boolean_query_result is not None:
                return boolean_query_result.get_from_corpus(self._corpus)
            else:
                return boolean_query_result
        elif '"' in query:
            # Handle phrase query
            phrase_query_result = self.handle_phrase_query(query)
            if phrase_query_result is not None:
                return phrase_query_result.get_from_corpus(self._corpus)
            else:
                print('No matches found')
        else:
            print(f"Query {query} can't be processed.")

    # Method that handles a phrase query by tokenizing the query, retrieving posting lists for each term, and finding consecutive postings within the lists
    def handle_phrase_query(self, query):
        tokens = normalize_stem_tokenize(query)
        posting_lists = []
        for term in tokens:
            posting_lists.append(self._index[Term(term, PostingList())])
        if len(posting_lists) == 1:
            return posting_lists[0]
        elif len(posting_lists) == 2:
            result = consecutive(posting_lists[0], posting_lists[1])
        else:
            result = consecutive(posting_lists[0], posting_lists[1])
            if result is None:
                return result
            for i in range(2,len(posting_lists)):
                result = consecutive(result, posting_lists[i])
                if result is None:
                    return result
        return result

    # Method that handles a boolean query by parsing, evaluating, and executing the boolean expressions within the query
    def handle_boolean_query(self, query):
        parsed_query = parse_query(query)
        expressions = evaluate_expression(parsed_query)
        results = {}
        for item in expressions.items():
            expr = item
            if item[1][0] in ['AND','OR']:
                for term in item[1][1]:
                    if term in results:
                        expr[1][1][item[1][1].index(term)] = results[term]
                results[item[0]] = self.boolean_query(expr[1])
            elif item[1][0] == 'NOT':
                if item[1][1] in results:
                    expr = (expr[0],(expr[1][0],results[item[1][1]]))
                results[item[0]] = self.boolean_query(expr[1])
        return results[item[0]]

    # Method executes a single boolean query expression, including AND, OR, and NOT operations, utilizing the dynamic index's posting lists
    def boolean_query(self, expression):
        if expression[0] == 'AND':
            posting_lists = []
            for term in expression[1]:
                if isinstance(term, str):
                    pl = self._index[Term(term, PostingList())]
                    if pl is not None:
                        posting_lists.append(pl)
                    else:
                        return None
                elif isinstance(term, PostingList):
                    posting_lists.append(term)
            result = posting_lists[0]
            for posting_list in posting_lists[1:]:
                result = result.intersection(posting_list)
        elif expression[0] == 'OR':
            posting_lists = []
            for term in expression[1]:
                if isinstance(term, str):
                    pl = self._index[Term(term, PostingList())]
                    if pl is not None:
                        posting_lists.append(pl)
                elif isinstance(term, PostingList):
                    posting_lists.append(term)
            if len(posting_lists) > 1:
                result = posting_lists[0]
                for posting_list in posting_lists[1:]:
                    result = result.union(posting_list)
            elif len(posting_lists) == 1:
                result = posting_lists[0]
            else:
                result = None
        elif expression[0] == 'NOT':
            if isinstance(expression[1], str):
                term_posting_list = self._index[Term(expression[1], PostingList())]
                if term_posting_list is None:
                    return PostingList.from_posting_list([Posting(docID, []) for docID in set(range(len(self._corpus)))])
            elif isinstance(expression[1], PostingList):
                term_posting_list = expression[1]
            all_documents = set(range(len(self._corpus)))
            term_documents = set([posting._docID for posting in term_posting_list._postings])
            not_documents = all_documents - term_documents
            result = PostingList.from_posting_list([Posting(docID, []) for docID in not_documents])
        return result

    def __repr__(self):
        return f"IR system:\nCorpus of {len(self._corpus)} elements.\nIndex: {self._index}"



""" testing """

# pass as argument "upload" if you want to upload an already existing IR system
# don't pass anything if you want to build an IR system indexing a corpus of documents
if len(sys.argv) > 1:
    user_input = str(sys.argv[1])
else:
    user_input = None

if user_input != "upload":
    corpus = read_movie_descriptions()
    
    print("The corpus is divided in 3 parts:")
    print(" - A contains corpus[:25000]")
    print(" - B contains corpus[25000:36000]")
    print(" - C contains corpus[36000:]")
    print('\n--------------------------------------------------------------------------------------\n')
    
    print("The initial IR system is built indexing A and B.")
    mini_ir = IRsystem.from_corpus(corpus[:36000],main_index_order = 100, aux_index_order = 25)
    
    print(mini_ir)
    print('\n')
    
    ir_filename = "my_ir_AB.pkl"
    with open(ir_filename, "wb") as ir_file:
        pickle.dump(mini_ir, ir_file)
        
    print("Initial IR system (A+B) saved to", ir_filename)
    print('\n--------------------------------------------------------------------------------------\n')
    
    print("Part C of the corpus is added to the IR system.")
    start_time = time.perf_counter()
    for i, document in enumerate(corpus[36000:]):
        mini_ir.add_document(document)
        if (i % 500 == 0):
            mid_time = time.perf_counter()
            elapsed_time = mid_time - start_time
            logging.info(f'After {elapsed_time}s, {i} inserted')
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
    
    print(mini_ir)
    print('\n')
    
    ir_filename = "my_ir_ABC.pkl"
    with open(ir_filename, "wb") as ir_file:
        pickle.dump(mini_ir, ir_file)
    
    print("Intermediate IR system (A+B+C) saved to", ir_filename)
    print('\n--------------------------------------------------------------------------------------\n')
    
    print("Part B of the corpus is removed from the IR system.")
    for document in corpus[25000:36000]:
        mini_ir.delete_document(document.title)
    
    print(mini_ir)
    print("\n")
    
    ir_filename = "my_ir_AC.pkl"
    with open(ir_filename, "wb") as ir_file:
        pickle.dump(mini_ir, ir_file)
    
    print("Final IR system (A+C) saved to", ir_filename)
    print('\n--------------------------------------------------------------------------------------\n')
    
    print("Perform fast merge on the final IR system.")
    mini_ir.fast_merge_dynamic_index()
    print(mini_ir)
    
    ir_filename = "my_ir_AC_fast_merge.pkl"
    with open(ir_filename, "wb") as ir_file:
        pickle.dump(mini_ir, ir_file)
        
    print("Final IR system (A+C) with fast merge saved to", ir_filename)

else:
    # Load the index from file
    ir_filename = input("Enter the exact filename of the IR system to upload: ")
    try:
        print("Loading ",ir_filename, "...")
        with open(ir_filename, "rb") as ir_file:
            mini_ir = pickle.load(ir_file)
    except FileNotFoundError:
        print("The specified file does not exist!")    

    # Check the type to see if the file loaded is actually an IR system
    print("The loaded file is of type: ",type(mini_ir))
    print("Some info about the loaded file:\n",mini_ir)
    print('\n--------------------------------------------------------------------------------------\n')
    
    while True:
        print('Rules for valid queries:')
        print(' - Prase queries must be enclosed in ""')
        print(' - If you need to search one single word, enclose it in "" (like a phrase query)')
        print(' - Boolean queries must contain at least one boolean operator (AND,OR, NOT) and may contain parenthesis. They must not contain "\n\n ')
        query = input('Enter a valid query or press Enter to exit:')
        if query == "":
            break
        start_time = time.perf_counter()
        answ = mini_ir.answer_query(query)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print('\n')
        print("Query execution time: ", elapsed_time)
        print('\n')
        print('Answer:\n')
        print(answ)
        print('\n--------------------------------------------------------------------------------------\n')

    print('IR system closed.')
