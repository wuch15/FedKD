from six.moves.urllib.parse import urlparse
from collections import Counter
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import re
from utils import word_tokenize


def get_domain(url):
    domain = urlparse(url).netloc
    return domain

def read_news_bert(news_path, args, tokenizer, mode='train'):
    news = {}
    categories = []
    subcategories = []
    domains = []
    news_index = {}
    index = 1

    with tf.io.gfile.GFile(news_path, "r") as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title, abstract, body, _, _ = splited
            url=""
            news_index[doc_id] = index
            index += 1

            if 'title' in args.news_attributes:
                title = title.lower()#+body.lower()[:2000]#xxx
                title = tokenizer(title, max_length=args.num_words_title, \
                pad_to_max_length=True, truncation=True)
            else:
                title = []

            if 'abstract' in args.news_attributes:
                abstract = abstract.lower()
                abstract = tokenizer(abstract, max_length=args.num_words_abstract, \
                pad_to_max_length=True, truncation=True)
            else:
                abstract = []

            if 'body' in args.news_attributes:
                body = body.lower()[:2000]
                body = tokenizer(body, max_length=args.num_words_body, \
                pad_to_max_length=True, truncation=True)
            else:
                body = []

            #if 'category' in args.news_attributes:
            categories.append(category)
            #else:
            #    category = None
            
            #if 'subcategory' in args.news_attributes:
            subcategories.append(subcategory)
            #else:
            #    subcategory = None

            if 'domain' in args.news_attributes:
                domain = get_domain(url)
                domains.append(domain)
            else:
                domain = None

            news[doc_id] = [title, abstract, body, category, domain, subcategory]
            if index>=2:
                break

    if mode == 'train':
        #categories = list(set(categories))
        category_dict = {}
        index = 1
        for x in categories:
            if x not in category_dict:
                category_dict[x] = index
                index += 1

        #subcategories = list(set(subcategories))
        subcategory_dict = {}
        index = 1
        for x in subcategories:
            if x not in subcategory_dict:
                subcategory_dict[x] = index
                index += 1

        #domains = list(set(domains))
        domain_dict = {}
        index = 1
        for x in domains:
            if x not in domain_dict:
                domain_dict[x] = index
                index += 1

        return news, news_index, category_dict, domain_dict, subcategory_dict
    elif mode == 'test':
        return news, news_index
    else:
        assert False, 'Wrong mode!'

def read_news(news_path, args, mode='train'):
    news = {}
    categories = []
    subcategories = []
    domains = []
    news_index = {}
    index = 1
    word_cnt = Counter()

    with tf.io.gfile.GFile(news_path, "r") as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title, abstract, url, _, _ = splited
            body = ""
            news_index[doc_id] = index
            index += 1

            if 'title' in args.news_attributes:
                title = title.lower()#+body.lower()[:2000]
                title = word_tokenize(title)
            else:
                title = []

            if 'abstract' in args.news_attributes:
                abstract = abstract.lower()
                abstract = word_tokenize(abstract)
            else:
                abstract = []

            if 'body' in args.news_attributes:
                body = body.lower()[:2000]
                body = word_tokenize(body)
            else:
                body = []

            if 'category' in args.news_attributes:
                categories.append(category)
            else:
                category = None
            
            if 'subcategory' in args.news_attributes:
                subcategories.append(subcategory)
            else:
                subcategory = None

            if 'domain' in args.news_attributes:
                domain = get_domain(url)
                domains.append(domain)
            else:
                domain = None

            news[doc_id] = [title, abstract, body, category, domain, subcategory]
            if mode == 'train':
                word_cnt.update(title + abstract + body)

    if mode == 'train':
        word = [k for k, v in word_cnt.items() if v > args.filter_num]
        word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
        categories = list(set(categories))
        category_dict = {}
        index = 1
        for x in categories:
            category_dict[x] = index
            index += 1

        subcategories = list(set(subcategories))
        subcategory_dict = {}
        index = 1
        for x in subcategories:
            subcategory_dict[x] = index
            index += 1

        domains = list(set(domains))
        domain_dict = {}
        index = 1
        for x in domains:
            domain_dict[x] = index
            index += 1

        return news, news_index, category_dict, word_dict, domain_dict, subcategory_dict
    elif mode == 'test':
        return news, news_index
    else:
        assert False, 'Wrong mode!'


def get_doc_input(news, news_index, category_dict, word_dict, domain_dict,
                  subcategory_dict, args):
    news_num = len(news) + 1
    if 'title' in args.news_attributes:
        news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
    else:
        news_title = None

    if 'abstract' in args.news_attributes:
        news_abstract = np.zeros((news_num, args.num_words_abstract),
                                 dtype='int32')
    else:
        news_abstract = None

    if 'body' in args.news_attributes:
        news_body = np.zeros((news_num, args.num_words_body), dtype='int32')
    else:
        news_body = None

    if 'category' in args.news_attributes:
        news_category = np.zeros((news_num, 1), dtype='int32')
    else:
        news_category = None

    if 'domain' in args.news_attributes:
        news_domain = np.zeros((news_num, 1), dtype='int32')
    else:
        news_domain = None

    if 'subcategory' in args.news_attributes:
        news_subcategory = np.zeros((news_num, 1), dtype='int32')
    else:
        news_subcategory = None

    for key in tqdm(news):
        title, abstract, body, category, domain, subcategory = news[key]
        doc_index = news_index[key]

        if 'title' in args.news_attributes:
            for word_id in range(min(args.num_words_title, len(title))):
                if title[word_id] in word_dict:
                    news_title[doc_index,
                               word_id] = word_dict[title[word_id].lower()]

        if 'abstract' in args.news_attributes:
            for word_id in range(min(args.num_words_abstract, len(abstract))):
                if abstract[word_id] in word_dict:
                    news_abstract[doc_index, word_id] = word_dict[
                        abstract[word_id].lower()]

        if 'body' in args.news_attributes:
            for word_id in range(min(args.num_words_body, len(body))):
                if body[word_id] in word_dict:
                    news_body[doc_index,
                              word_id] = word_dict[body[word_id].lower()]

        if 'category' in args.news_attributes:
            news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        
        if 'subcategory' in args.news_attributes:
            news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0
        
        if 'domain' in args.news_attributes:
            news_domain[doc_index, 0] = domain_dict[domain] if domain in domain_dict else 0

    return news_title, news_abstract, news_body, news_category, news_domain, news_subcategory

def get_doc_input_bert(news, news_index, category_dict, domain_dict, subcategory_dict, args):
    news_num = len(news) + 1
    if 'title' in args.news_attributes:
        news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
        news_title_type = np.zeros((news_num, args.num_words_title), dtype='int32')
        news_title_attmask = np.zeros((news_num, args.num_words_title), dtype='int32')
    else:
        news_title = None
        news_title_type = None
        news_title_attmask = None

    if 'abstract' in args.news_attributes:
        news_abstract = np.zeros((news_num, args.num_words_abstract), dtype='int32')
        news_abstract_type = np.zeros((news_num, args.num_words_abstract), dtype='int32')
        news_abstract_attmask = np.zeros((news_num, args.num_words_abstract), dtype='int32')
    else:
        news_abstract = None
        news_abstract_type = None
        news_abstract_attmask = None

    if 'body' in args.news_attributes:
        news_body = np.zeros((news_num, args.num_words_body), dtype='int32')
        news_body_type = np.zeros((news_num, args.num_words_body), dtype='int32')
        news_body_attmask = np.zeros((news_num, args.num_words_body), dtype='int32')
    else:
        news_body = None
        news_body_type = None
        news_body_attmask = None

    #if 'category' in args.news_attributes:
    news_category = np.zeros((news_num, 1), dtype='int32')
    #else:
    #    news_category = None

    if 'domain' in args.news_attributes:
        news_domain = np.zeros((news_num, 1), dtype='int32')
    else:
        news_domain = None

    #if 'subcategory' in args.news_attributes:
    news_subcategory = np.zeros((news_num, 1), dtype='int32')
    #else:
    #    news_subcategory = None

    for key in tqdm(news):
        title, abstract, body, category, domain, subcategory = news[key]
        doc_index = news_index[key]

        if 'title' in args.news_attributes:
            news_title[doc_index] = title['input_ids']
            news_title_type[doc_index] = title['token_type_ids']
            news_title_attmask[doc_index] = title['attention_mask']          

        if 'abstract' in args.news_attributes:
            news_abstract[doc_index] = abstract['input_ids']
            news_abstract_type[doc_index] = abstract['token_type_ids']
            news_abstract_attmask[doc_index] = abstract['attention_mask']

        if 'body' in args.news_attributes:
            news_body[doc_index] = body['input_ids']
            news_body_type[doc_index] = body['token_type_ids']
            news_body_attmask[doc_index] = body['attention_mask']

        #if 'category' in args.news_attributes:
        news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        
        #if 'subcategory' in args.news_attributes:
        news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0
        
        if 'domain' in args.news_attributes:
            news_domain[doc_index, 0] = domain_dict[domain] if domain in domain_dict else 0

    return news_title, news_title_type, news_title_attmask, \
           news_abstract, news_abstract_type, news_abstract_attmask, \
           news_body, news_body_type, news_body_attmask, \
           news_category, news_domain, news_subcategory

if __name__ == "__main__":
    from parameters import parse_args
    args = parse_args()
    args.news_attributes = ['title', 'body', 'category', 'subcategory', 'domain']
    news, news_index, category_dict, word_dict, domain_dict, subcategory_dict = read_news(
        "../MIND/train/news.tsv",
        args)
    news_title, news_abstract, news_body, news_category, news_domain, news_subcategory = get_doc_input(
        news, news_index, category_dict, word_dict, domain_dict, subcategory_dict, args)

    print(category_dict)
    print(news_category)