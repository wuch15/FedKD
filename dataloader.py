import sys
import traceback
import logging
import time
import random
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import IterableDataset
from streaming import StreamSampler, StreamSamplerTest
import utils
import pickle


def news_sample(news, ratio):
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


class DataLoaderTrain(IterableDataset):
    def __init__(self,
                 data_dir,
                 filename_pat,
                 args,
                 world_size,
                 worker_rank,
                 cuda_device_idx,
                 news_index,
                 news_combined,
                 word_dict,
                 enable_prefetch=True,
                 enable_shuffle=False,
                 enable_gpu=True):
        self.data_dir = data_dir
        self.filename_pat = filename_pat

        self.npratio = args.npratio
        self.user_log_length = args.user_log_length
        self.batch_size = args.batch_size

        self.worker_rank = worker_rank
        self.world_size = world_size
        self.cuda_device_idx = cuda_device_idx
        # data loader only cares about the config after tokenization.
        self.sampler = None

        self.process_uet = args.process_uet
        self.process_bing = args.process_bing
        self.user_uet_length = args.user_uet_length
        self.num_words_uet = args.num_words_uet
        self.user_bing_length = args.user_bing_length
        self.num_words_bing = args.num_words_bing
        self.shuffle_buffer_size = args.shuffle_buffer_size

        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = enable_gpu
        self.epoch = -1

        self.news_combined = news_combined
        randomindex=np.arange(len(news_combined))
        np.random.shuffle(randomindex)
        self.news_random_index = randomindex
        self.news_index = news_index
        self.word_dict = word_dict
        self.news_current_index=0

    def start(self):
        self.epoch += 1
        self.sampler = StreamSampler(
            data_dir=self.data_dir,
            filename_pat=self.filename_pat,
            batch_size=self.batch_size,
            worker_rank=self.worker_rank,
            world_size=self.world_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_buffer_size=self.shuffle_buffer_size,
            shuffle_seed=self.epoch,  # epoch id as shuffle random seed
        )
        self.sampler.__iter__()

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length-len(x)) + x[-fix_length:]
            mask = [0] * (fix_length-len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[:fix_length] + [padding_value]*(fix_length-len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (len(x) - fix_length)
        return pad_x, mask

    def _produce(self):
        # need to reset cuda device in produce thread.
        if self.enable_gpu:
            torch.cuda.set_device(self.cuda_device_idx)
        try:
            self.epoch += 1
            self.sampler = StreamSampler(
                data_dir=self.data_dir,
                filename_pat=self.filename_pat,
                batch_size=self.batch_size,
                worker_rank=self.worker_rank,
                world_size=self.world_size,
                enable_shuffle=self.enable_shuffle,
                shuffle_seed=self.epoch,  # epoch id as shuffle random seed
            )
            # t0 = time.time()
            for batch in self.sampler:
                if self.stopped:
                    break
                context = self._process(batch)
                self.outputs.put(context)
                self.aval_count += 1
                # logging.info(f"_produce cost:{time.time()-t0}")
                # t0 = time.time()
        except:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    def start_async(self):
        self.aval_count = 0
        self.stopped = False
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def parse_sent(self, sent, fix_length):
        sent = [self.word_dict[w] if w in self.word_dict else 0 for w in utils.word_tokenize(sent)]
        sent, _ = self.pad_to_fix_len(sent, fix_length, padding_front=False)
        return sent

    def parse_sents(self, sents, max_sents_num, max_sent_length, padding_front=True):
        sents, sents_mask = self.pad_to_fix_len(sents, max_sents_num, padding_value='')
        sents = [self.parse_sent(s, max_sent_length) for s in sents]
        sents = np.stack(sents, axis=0)
        sents_mask = np.array(sents_mask)
        return sents, sents_mask

    def _parse_uet(self, uets):
        uets = uets.split('#TAB#')
        user_uet_feature, uet_log_mask = self.parse_sents(uets, self.user_uet_length, self.num_words_uet)
        return user_uet_feature, uet_log_mask

    def _parse_bing(self, bings):
        bings = [' '.join(i.split("#N#")) for i in bings.split('#TAB#')]
        user_bing_feature, bing_log_mask = self.parse_sents(bings, self.user_bing_length, self.num_words_bing)
        return user_bing_feature, bing_log_mask

    def _process(self, batch):
        batch_size = len(batch)
        #print(batch)
        batch_poss, batch = batch
        batch_poss = [x.decode(encoding="utf-8") for x in batch_poss]
        batch = [x.decode(encoding="utf-8").split("\t") for x in batch]
        label = 0
        user_feature_batch, log_mask_batch, news_feature_batch, label_batch = [], [], [], []
        
        user_uet_feature_batch, uet_log_mask_batch = [], []
        user_bing_feature_batch, bing_log_mask_batch = [], []
        user_id_batch=[]
        random_news_batch=[]
        cate_label_batch=[]
        for poss, line in zip(batch_poss, batch):
            click_docs = line[3].split()
            if self.process_uet:
                user_uets = line[-3]
                user_uet_feature, uet_log_mask = self._parse_uet(user_uets)
                user_uet_feature_batch.append(user_uet_feature)
                uet_log_mask_batch.append(uet_log_mask)


            if self.process_bing:
                user_bings = line[-5]
                user_bing_feature, bing_log_mask = self._parse_bing(user_bings)
                user_bing_feature_batch.append(user_bing_feature)
                bing_log_mask_batch.append(bing_log_mask)

            click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(click_docs),
                                             self.user_log_length)

            user_feature = self.news_combined[click_docs]
            random_news_batch.append(self.news_combined[self.news_random_index[self.news_current_index]])
            cate_label_batch.append(self.news_combined[self.news_random_index[self.news_current_index]][-2])
            self.news_current_index=(self.news_current_index+1)%len(self.news_combined)
            sess_news = [i.split('-') for i in line[4].split()]
            sess_neg = [i[0] for i in sess_news if i[-1] == '0']
        
            poss = self.trans_to_nindex([poss])
            sess_neg = self.trans_to_nindex(sess_neg)

            if len(sess_neg) > 0:
                neg_index = news_sample(list(range(len(sess_neg))),
                                        self.npratio)
                sam_negs = [sess_neg[i] for i in neg_index]
            else:
                sam_negs = [0] * self.npratio
            sample_news = poss + sam_negs

            news_feature = self.news_combined[sample_news]
            user_feature_batch.append(user_feature)
            log_mask_batch.append(log_mask)
            news_feature_batch.append(news_feature)
            label_batch.append(label)
            user_id_batch.append([0])

        if self.enable_gpu:
            user_feature_batch = torch.LongTensor(user_feature_batch).cuda()
            log_mask_batch = torch.FloatTensor(log_mask_batch).cuda()
            news_feature_batch = torch.LongTensor(news_feature_batch).cuda()
            label_batch = torch.LongTensor(label_batch).cuda()
            user_id_batch = torch.LongTensor(user_id_batch).cuda()
            user_uet_feature_batch = torch.LongTensor(user_uet_feature_batch).cuda()
            uet_log_mask_batch = torch.FloatTensor(uet_log_mask_batch).cuda()
            random_news_batch = torch.LongTensor(random_news_batch).cuda()
            cate_label_batch = torch.LongTensor(cate_label_batch).cuda()
            user_bing_feature_batch = torch.LongTensor(user_bing_feature_batch).cuda()
            bing_log_mask_batch = torch.FloatTensor(bing_log_mask_batch).cuda()
        else:
            user_feature_batch = torch.LongTensor(user_feature_batch)
            log_mask_batch = torch.FloatTensor(log_mask_batch)
            news_feature_batch = torch.LongTensor(news_feature_batch)
            label_batch = torch.LongTensor(label_batch)
            user_id_batch = torch.LongTensor(user_id_batch)
            random_news_batch = torch.LongTensor(random_news_batch)
            cate_label_batch = torch.LongTensor(cate_label_batch)
            user_uet_feature_batch = torch.LongTensor(user_uet_feature_batch)
            uet_log_mask_batch = torch.FloatTensor(uet_log_mask_batch)
                
            user_bing_feature_batch = torch.LongTensor(user_bing_feature_batch)
            bing_log_mask_batch = torch.FloatTensor(bing_log_mask_batch)

        return user_feature_batch, log_mask_batch, news_feature_batch, label_batch, user_uet_feature_batch, uet_log_mask_batch, user_bing_feature_batch, bing_log_mask_batch, user_id_batch, random_news_batch, cate_label_batch

    def __iter__(self):
        """Implement IterableDataset method to provide data iterator."""
        logging.info("DataLoader __iter__()")
        if self.enable_prefetch:
            self.join()
            self.start_async()
        else:
            self.start()
        return self

    def __next__(self):
        if self.sampler and self.sampler.reach_end() and self.aval_count == 0:
            raise StopIteration
        if self.enable_prefetch:
            next_batch = self.outputs.get()
            self.outputs.task_done()
            self.aval_count -= 1
        else:
            next_batch = self._process(self.sampler.__next__())
        return next_batch

    def join(self):
        self.stopped = True
        if self.sampler:
            if self.enable_prefetch:
                while self.outputs.qsize() > 0:
                    self.outputs.get()
                    self.outputs.task_done()
                self.outputs.join()
                self.pool.shutdown(wait=True)
                logging.info("shut down pool.")
            self.sampler = None


class DataLoaderTest(DataLoaderTrain):
    def __init__(self,
                 data_dir,
                 filename_pat,
                 args,
                 world_size,
                 worker_rank,
                 cuda_device_idx,
                 news_index,
                 news_scoring,
                 word_dict,
                 news_bias_scoring=None,
                 enable_prefetch=True,
                 enable_shuffle=False,
                 enable_gpu=True):
        self.data_dir = data_dir
        self.filename_pat = filename_pat

        self.npratio = args.npratio
        self.user_log_length = args.user_log_length
        self.batch_size = args.batch_size

        self.worker_rank = worker_rank
        self.world_size = world_size
        self.cuda_device_idx = cuda_device_idx
        # data loader only cares about the config after tokenization.
        self.sampler = None

        self.process_uet = args.process_uet
        self.process_bing = args.process_bing
        self.user_uet_length = args.user_uet_length
        self.num_words_uet = args.num_words_uet
        self.user_bing_length = args.user_bing_length
        self.num_words_bing = args.num_words_bing

        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = enable_gpu
        self.epoch = -1

        self.news_scoring = news_scoring
        self.news_bias_scoring = news_bias_scoring
        self.news_index = news_index
        self.word_dict = word_dict

    def start(self):
        self.epoch += 1
        self.sampler = StreamSamplerTest(
            data_dir=self.data_dir,
            filename_pat=self.filename_pat,
            batch_size=self.batch_size,
            worker_rank=self.worker_rank,
            world_size=self.world_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_seed=self.epoch,  # epoch id as shuffle random seed
        )
        self.sampler.__iter__()

    def _produce(self):
        # need to reset cuda device in produce thread.
        if self.enable_gpu:
            torch.cuda.set_device(self.cuda_device_idx)
        try:
            self.epoch += 1
            self.sampler = StreamSamplerTest(
                data_dir=self.data_dir,
                filename_pat=self.filename_pat,
                batch_size=self.batch_size,
                worker_rank=self.worker_rank,
                world_size=self.world_size,
                enable_shuffle=self.enable_shuffle,
                shuffle_seed=self.epoch,  # epoch id as shuffle random seed
            )
            # t0 = time.time()
            for batch in self.sampler:
                if self.stopped:
                    break
                context = self._process(batch)
                self.outputs.put(context)
                self.aval_count += 1
                # logging.info(f"_produce cost:{time.time()-t0}")
                # t0 = time.time()
        except:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    def _process(self, batch):
        batch_size = len(batch)
        batch = [x.decode(encoding="utf-8").split("\t") for x in batch]

        user_feature_batch, log_mask_batch, news_feature_batch, news_bias_batch, label_batch = [], [], [], [], []

        user_uet_feature_batch, uet_log_mask_batch = [], []
        user_bing_feature_batch, bing_log_mask_batch = [], []
        user_id_batch=[]
        for line in batch:
            click_docs = line[3].split()
            if self.process_uet:
                user_uets = line[-3]
                user_uet_feature, uet_log_mask = self._parse_uet(user_uets)
                user_uet_feature_batch.append(user_uet_feature)
                uet_log_mask_batch.append(uet_log_mask)

            if self.process_bing:
                user_bings = line[-5]
                user_bing_feature, bing_log_mask = self._parse_bing(user_bings)
                user_bing_feature_batch.append(user_bing_feature)
                bing_log_mask_batch.append(bing_log_mask)

            click_docs, log_mask  = self.pad_to_fix_len(self.trans_to_nindex(click_docs),
                                             self.user_log_length)
            user_feature = self.news_scoring[click_docs]

            sample_news = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
            labels = [int(i.split('-')[1]) for i in line[4].split()]

            news_feature = self.news_scoring[sample_news]
            if self.news_bias_scoring is not None:
                news_bias = self.news_bias_scoring[sample_news]
            else:
                news_bias = [0] * len(sample_news)
            user_feature_batch.append(user_feature)
            log_mask_batch.append(log_mask)
            news_feature_batch.append(news_feature)
            news_bias_batch.append(news_bias)
            label_batch.append(np.array(labels))
            user_id_batch.append([0])

        if self.enable_gpu:
            user_feature_batch = torch.FloatTensor(user_feature_batch).cuda()
            log_mask_batch = torch.FloatTensor(log_mask_batch).cuda()
            user_id_batch= torch.LongTensor(user_id_batch).cuda()
            user_uet_feature_batch = torch.LongTensor(user_uet_feature_batch).cuda()
            uet_log_mask_batch = torch.FloatTensor(uet_log_mask_batch).cuda()

            user_bing_feature_batch = torch.LongTensor(user_bing_feature_batch).cuda()
            bing_log_mask_batch = torch.FloatTensor(bing_log_mask_batch).cuda()
            
        else:
            user_feature_batch = torch.FloatTensor(user_feature_batch)
            log_mask_batch = torch.FloatTensor(log_mask_batch)
            user_id_batch= torch.LongTensor(user_id_batch)
            user_uet_feature_batch = torch.LongTensor(user_uet_feature_batch)
            uet_log_mask_batch = torch.FloatTensor(uet_log_mask_batch)
            
            user_bing_feature_batch = torch.LongTensor(user_bing_feature_batch)
            bing_log_mask_batch = torch.FloatTensor(bing_log_mask_batch)

        return user_feature_batch, log_mask_batch, news_feature_batch, news_bias_batch, label_batch, user_uet_feature_batch, uet_log_mask_batch, user_bing_feature_batch, bing_log_mask_batch, user_id_batch


def test_load(args, news_index, news_combined, word_dict):
    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(True)
    dataloader = DataLoaderTrain(
        data_dir=
        "../MIND/train/",
        filename_pat="behaviors_*.tsv",
        args=args,
        world_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        news_index=news_index,
        word_dict=word_dict, 
        news_combined=news_combined,
        enable_prefetch=False,
        enable_shuffle=True
    )

    t0 = time.time()
    # torch data loader warper will remove the benefit brought by prefetch.
    # dataloader = torch.utils.data.DataLoader(dataloader)
    for i_epoch in range(3):
        for i_batch, context in enumerate(dataloader):
            if i_batch > 100:
                break
            time.sleep(0.2)
            t = time.time() - t0
            # for c in context:
            #     print(c)
            print(context[2])
            time.sleep(15)
            logging.info(
                f"epoch:{i_epoch}, batch:{i_batch}, load: {t-0.2}, total:{t}")
            t0 = time.time()
    dataloader.join()


def test_load_test(args, news_index, news_scoring, word_dict):
    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(True)
    dataloader = DataLoaderTest(
        data_dir=
        "../MIND/train/",
        filename_pat="behaviors_*.tsv",
        args=args,
        world_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        news_index=news_index,
        word_dict=word_dict, 
        news_scoring=news_scoring,
        enable_prefetch=True,
        enable_shuffle=True
    )

    t0 = time.time()
    # torch data loader warper will remove the benefit brought by prefetch.
    # dataloader = torch.utils.data.DataLoader(dataloader)
    for i_epoch in range(3):
        for i_batch, context in enumerate(dataloader):
            if i_batch > 10:
                break
            time.sleep(0.2)
            t = time.time() - t0
            print(context)
            logging.info(
                f"epoch:{i_epoch}, batch:{i_batch}, load: {t-0.2}, total:{t}")
            t0 = time.time()
    dataloader.join()


if __name__ == "__main__":
    from preprocess import read_news_bert, get_doc_input_bert
    from parameters import parse_args
    from tnlrv3.modeling import TuringNLRv3ForSequenceClassification
    from tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
    from tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer

    from news_word_embedding import infer_news

    MODEL_CLASSES = {
        'tnlrv3': (TuringNLRv3Config, TuringNLRv3ForSequenceClassification, TuringNLRv3Tokenizer),
    }
    args = parse_args()
    args.npratio = 4
    args.batch_size = 64
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    bert_model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config)
                                    
    news, news_index, category_dict, domain_dict, subcategory_dict = read_news_bert(
        "../MIND/train/news.tsv",
        args,
        tokenizer
        )
    news_title, news_title_type, news_title_attmask, \
        news_abstract, news_abstract_type, news_abstract_attmask, \
        news_body, news_body_type, news_body_attmask, \
        news_category, news_domain, news_subcategory  = get_doc_input_bert(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)
    utils.setuplogger()
    news_combined = np.concatenate([
        x for x in
        [news_title, news_title_type, news_title_attmask, \
            news_abstract, news_abstract_type, news_abstract_attmask, \
            news_body, news_body_type, news_body_attmask, \
            news_category, news_domain, news_subcategory]
        if x is not None], axis=1)

    word_dict=None
    #test_load(args, news_index, news_combined, word_dict)
    test_load_test(args, news_index, news_combined, word_dict)
