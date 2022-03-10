import numpy as np
import torch
import pickle
import hashlib
import logging
from tqdm.auto import tqdm
import torch.optim as optim
from pathlib import Path
import utils
import os
import sys
import logging
from dataloader import DataLoaderTrain, DataLoaderTest
from torch.utils.data import Dataset, DataLoader

from preprocess import read_news, read_news_bert, get_doc_input, get_doc_input_bert
from model import Model
from model_bert import ModelBert
from parameters import parse_args


from tnlrv3.modeling import TuringNLRv3ForSequenceClassification
from tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
from tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer


MODEL_CLASSES = {
    'tnlrv3': (TuringNLRv3Config, TuringNLRv3ForSequenceClassification, TuringNLRv3Tokenizer),
}

def train(args):
    assert args.enable_hvd  
    if args.enable_hvd:
        import horovod.torch as hvd

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(args.model_dir)


    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)
    print(args.news_attributes)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,output_hidden_states = True)
    config.num_hidden_layers = 12
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    config_student = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,output_hidden_states = True)
    config_student.num_hidden_layers = 4
    
    #assume that there are K clients
    teacher_bert_models=[]
    student_bert_models=[]
    client_num=4
    for i in range(client_num):
        bert_model_stu = model_class.from_pretrained(args.model_name_or_path,from_tf=bool('.ckpt' in args.model_name_or_path),config=config_student)
        bert_model_tea = model_class.from_pretrained(args.model_name_or_path,from_tf=bool('.ckpt' in args.model_name_or_path),config=config)
        for name,para in bert_model_stu.named_parameters():
            if 'embeddings' in name:
                para.requires_grad=False
        for name,para in bert_model_tea.named_parameters():
            if 'embeddings' in name:
                para.requires_grad=False                
        teacher_bert_models.append(bert_model_tea)
        student_bert_models.append(bert_model_stu)
    
    
    
    #preprocessing
    news, news_index, category_dict, domain_dict, subcategory_dict = read_news_bert(
        os.path.join(args.root_data_dir,
                    f'{args.market}/{args.train_date}/news.tsv'), 
        args,
        tokenizer
    )

    news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)

    news_combined = np.concatenate([
        x for x in
        [news_title, news_title_type, news_title_attmask, \
            news_abstract, news_abstract_type, news_abstract_attmask, \
            news_body, news_body_type, news_body_attmask, \
            news_category, news_domain, news_subcategory]
        if x is not None], axis=1)
    local_models=[]
    for i in range(client_num):
        local_client = ModelBert(args, teacher_bert_models[i], student_bert_models[i], len(category_dict), len(domain_dict), len(subcategory_dict))
        local_models.append(local_client.cuda())
    word_dict = None
    
    optimizers=[]
    for i in range(client_num):
        optimizer = optim.Adam([ {'params': local_models[i].parameters(), 'lr': 3e-6} ], lr=3e-6)
        optimizers.append(optimizer)

        if args.enable_hvd:
            hvd.broadcast_parameters(local_models[i].state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizers[i], root_rank=0)
            compression = hvd.Compression.none
            optimizers[i] = hvd.DistributedOptimizer(
                optimizers[i],
                named_parameters=local_models[i].named_parameters(),
                compression=compression,
                op=hvd.Average)


    dataloader = DataLoaderTrain(
        news_index=news_index,
        news_combined=news_combined,
        word_dict=word_dict,
        data_dir=os.path.join(args.root_data_dir,
                            f'{args.market}/{args.train_date}'),
        filename_pat=args.filename_pat,
        args=args,
        world_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=True,
        enable_shuffle=True,
        enable_gpu=args.enable_gpu,
    )
    latest_local_student_para=[{name:para.detach().cpu().numpy() for name,para in model.student_bert_models.named_parameters()} for model in local_models]

    logging.info('Local Training...')
    for comm_round in range(args.epochs):
        for i in range(client_num):
            loss = 0.0
            accuracy = 0.0

            for cnt, (log_ids, log_mask, input_ids, targets, 
            uet_ids, uet_mask, bing_ids, bing_mask, user_ids, random_news, catetarget) in enumerate(dataloader):
                #split dataset for different clients
                if cnt >= 10:
                    break
    
                if args.enable_gpu:
                    log_ids = log_ids.cuda(non_blocking=True)
                    log_mask = log_mask.cuda(non_blocking=True)
                    input_ids = input_ids.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)
                    uet_ids = uet_ids.cuda(non_blocking=True)
                    uet_mask = uet_mask.cuda(non_blocking=True)
                    bing_ids = bing_ids.cuda(non_blocking=True)
                    bing_mask = bing_mask.cuda(non_blocking=True)
                    user_ids = user_ids.cuda(non_blocking=True)
                    random_news = random_news.cuda(non_blocking=True)#not used
                    catetarget = catetarget.cuda(non_blocking=True)#not used
                
                bz_loss, y_hat = local_models[i](input_ids, log_ids, log_mask, targets, uet_ids, uet_mask, bing_ids, bing_mask, user_ids, random_news, catetarget)
                
                loss += bz_loss.data.float()
                accuracy += utils.acc(targets, y_hat)
            
                optimizers[i].zero_grad()
                bz_loss.backward()
                optimizers[i].step()
                logging.info(
                    '[{}] Ed: {}, loss: {:.5f}, acc: {:.5f}'.format(
                        hvd_rank, cnt * args.batch_size, loss.data / cnt,accuracy / (cnt)))
                    
                if hvd_rank == 0 and cnt % args.save_steps == 0:
                    ckpt_path = os.path.join(args.model_dir, f'epoch-{comm_round+1}-{cnt}.pt')
                    torch.save(
                        {
                            'model_state_dict': local_models[i].state_dict(),
                            'category_dict': category_dict,
                            'word_dict': word_dict,
                            'domain_dict': domain_dict,
                            'subcategory_dict': subcategory_dict
                        }, ckpt_path)
                    logging.info(f"Model saved to {ckpt_path}")
        energy=args.tmin+((1+comm_round)/args.epochs)*(args.tmax-args.tmin)
        all_compressed_update=[]
        for i in range(client_num): 
            student_update={}
            compressed_student_update={}

            for name in latest_local_student_para[i]:
                student_update[name]=local_models[i].student_bert_models.state_dict()[name].detach().cpu().numpy()-latest_local_student_para[i][name]
                if len(student_update[name].shape)>1 and 'embeddings' not in name:
                    u, sigma, v = np.linalg.svd(student_update[name], full_matrices=False)
                    print(name,u.shape,sigma.shape,v.shape)
                    threshold=0
                    if np.sum(np.square(sigma))==0:
                        compressed_student_update[name]=student_update[name]
                    else:
                        for singular_value_num in range(len(sigma)):
                            if np.sum(np.square(sigma[:singular_value_num]))>energy*np.sum(np.square(sigma)):
                                threshold=singular_value_num
                                break
                        u=u[:,:threshold]
                        sigma=sigma[:threshold]
                        v=v[:threshold,:]
                        compressed_student_update[name]=[u,sigma,v]
                elif 'embeddings' not in name:
                    compressed_student_update[name]=student_update[name]
                #assume that it is uploaded to server
                #*****
                #
            all_compressed_update.append(compressed_student_update)

        aggregated_para={name:[] for name in all_compressed_update[0]}
        for name in all_compressed_update[0]: 
            for i in range(client_num): 
                if len(all_compressed_update[i][name])==3:
                    
                    aggregated_para[name].append(np.dot(np.dot(all_compressed_update[i][name][0],all_compressed_update[i][name][1]),all_compressed_update[i][name][2]))
                else:
                    aggregated_para[name].append(all_compressed_update[i][name])
            aggregated_para[name]=np.mean(aggregated_para[name],axis=0)
        for name in aggregated_para:
            if len(aggregated_para[name].shape)>1 and 'embeddings' not in name:
                u, sigma, v = np.linalg.svd(aggregated_para[name], full_matrices=False)
                if np.sum(np.square(sigma))==0:
                    continue
                else:
                    threshold=0
                    for singular_value_num in range(len(sigma)):
                        if np.sum(np.square(sigma[:singular_value_num]))>=energy*np.sum(np.square(sigma)):
                            threshold=singular_value_num
                            break
                    u=u[:,:threshold]
                    sigma=sigma[:threshold]
                    v=v[:threshold,:]
                    aggregated_para[name]=[u,sigma,v]

        #assume that it is sent to clients
        #*****
        #      
        for name in aggregated_para:
            if len(aggregated_para[name])==3:
                aggregated_para[name]=np.dot(np.dot(aggregated_para[name][0],aggregated_para[name][1]),aggregated_para[name][2])
        for i in range(client_num): 
            for name in aggregated_para:
                for name2 in local_models[i].student_bert_models.state_dict():
                    if name in name2:
                        local_models[i].student_bert_models.state_dict()[name][:]=torch.FloatTensor(aggregated_para[name]).cuda()
            
    dataloader.join()


def test(args):

    if args.enable_hvd:
        import horovod.torch as hvd

    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    if args.load_ckpt_name is not None:
        #TODO: choose ckpt_path
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(args.model_dir)

    assert ckpt_path is not None, 'No ckpt found'
    checkpoint = torch.load(ckpt_path)

    if 'subcategory_dict' in checkpoint:
        subcategory_dict = checkpoint['subcategory_dict']
    else:
        subcategory_dict = {}

    category_dict = checkpoint['category_dict']
    word_dict = checkpoint['word_dict']
    domain_dict = checkpoint['domain_dict']
    # load model

    if args.apply_turing:
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,output_hidden_states = True)
        config.num_hidden_layers = 12
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                    do_lower_case=args.do_lower_case)
        bert_model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config)

        
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config2 = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,output_hidden_states = True)
        config.num_hidden_layers = 4
        bert_model2 = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config2)


        
        model = ModelBert(args, bert_model, bert_model2, len(category_dict), len(domain_dict), len(subcategory_dict))
    else:
        dummy_embedding_matrix = np.zeros(
            (len(word_dict) + 1, args.word_embedding_dim))
        model = Model(args, dummy_embedding_matrix, len(category_dict),
                    len(domain_dict), len(subcategory_dict))
                  
    if args.enable_gpu:
        model.cuda()

    model.load_state_dict(checkpoint['model_state_dict'],False)
    logging.info(f"Model loaded from {ckpt_path}")

    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    model.eval()
    torch.set_grad_enabled(False)

    if args.apply_turing:
        news, news_index, category_dict, domain_dict, subcategory_dict = read_news_bert(
            os.path.join(args.root_data_dir,
                        f'{args.market}/{args.test_date}/news.tsv'), 
            args,
            tokenizer
        )

        news_title, news_title_type, news_title_attmask, \
        news_abstract, news_abstract_type, news_abstract_attmask, \
        news_body, news_body_type, news_body_attmask, \
        news_category, news_domain, news_subcategory = get_doc_input_bert(
            news, news_index, category_dict, domain_dict, subcategory_dict, args)

        news_combined = np.concatenate([
        x for x in
        [news_title, news_title_type, news_title_attmask, \
        news_abstract, news_abstract_type, news_abstract_attmask, \
        news_body, news_body_type, news_body_attmask, \
        news_category, news_domain, news_subcategory]
        if x is not None], axis=1)
        

        class NewsDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, idx):
                return self.data[idx]

            def __len__(self):
                return self.data.shape[0]

        def news_collate_fn(arr):
            arr = torch.LongTensor(arr)
            return arr

        news_dataset = NewsDataset(news_combined)
        news_dataloader = DataLoader(news_dataset,
                                    batch_size=args.batch_size * 4,
                                    num_workers=args.num_workers,
                                    collate_fn=news_collate_fn)

        news_scoring = []
        news_bias_scoring = []
        with torch.no_grad():
            for input_ids in tqdm(news_dataloader):
                input_ids = input_ids.cuda()
                news_vec = model.teacher_bert_models(input_ids)[0]
                news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
                news_scoring.extend(news_vec)


        news_scoring = np.array(news_scoring)
        news_bias_scoring = np.array(news_bias_scoring)

        logging.info("news scoring num: {}".format(news_scoring.shape[0]))
 
    if args.apply_turing:
        dataloader = DataLoaderTest(
            news_index=news_index,
            news_scoring=news_scoring,
            word_dict=word_dict,
            news_bias_scoring=news_bias_scoring if args.debias else None,
            data_dir=os.path.join(args.root_data_dir,
                                f'{args.market}/{args.test_date}'),
            filename_pat=args.filename_pat,
            args=args,
            world_size=hvd_size,
            worker_rank=hvd_rank,
            cuda_device_idx=hvd_local_rank,
            enable_prefetch=True,
            enable_shuffle=False,
            enable_gpu=args.enable_gpu,
        )
    else:
        dataloader = DataLoaderTest(
            news_index=news_index,
            news_scoring=news_scoring,
            word_dict=word_dict,
            news_bias_scoring=news_bias_scoring if args.debias else None,
            data_dir=os.path.join(args.root_data_dir,
                                f'{args.market}/{args.test_date}'),
            filename_pat=args.filename_pat,
            args=args,
            world_size=hvd_size,
            worker_rank=hvd_rank,
            cuda_device_idx=hvd_local_rank,
            enable_prefetch=True,
            enable_shuffle=False,
            enable_gpu=args.enable_gpu,
        )

    from metrics import roc_auc_score, ndcg_score, mrr_score, ctr_score

    AUC = [[], []]
    MRR = [[], []]
    nDCG5 = [[], []]
    nDCG10 = [[], []]
    CTR1 = [[], []]
    CTR3 = [[], []]
    CTR5 = [[], []]
    CTR10 = [[], []]

    if args.debias:
        AUC_add_bias = [[], []]
        MRR_add_bias = [[], []]
        nDCG5_add_bias = [[], []]
        nDCG10_add_bias = [[], []]
        CTR1_add_bias = [[], []]
        CTR3_add_bias = [[], []]
        CTR5_add_bias = [[], []]
        CTR10_add_bias = [[], []]

    # only news score
    if args.process_uet or args.process_bing:
        AUC_on = [[], []]
        MRR_on = [[], []]
        nDCG5_on = [[], []]
        nDCG10_on = [[], []]
        CTR1_on = [[], []]
        CTR3_on = [[], []]
        CTR5_on = [[], []]
        CTR10_on = [[], []]


    def print_metrics(hvd_local_rank, cnt, x):
        logging.info("[{}] Ed: {}: {}".format(hvd_local_rank, cnt, \
            '\t'.join(["{:0.2f}".format(i * 100) for i in x])))

    def get_mean(arr):
        return [np.array(i).mean() for i in arr]

    #for cnt, (log_vecs, log_mask, news_vecs, news_bias, labels) in enumerate(dataloader):
    
    for cnt, (log_vecs, log_mask, news_vecs, news_bias, labels, \
        uet_ids, uet_mask, bing_ids, bing_mask, user_ids) in enumerate(dataloader):
        his_lens = torch.sum(log_mask, dim=-1).to(torch.device("cpu")).detach().numpy()

        if args.enable_gpu:
            log_vecs = log_vecs.cuda(non_blocking=True)
            log_mask = log_mask.cuda(non_blocking=True)
            uet_ids = uet_ids.cuda(non_blocking=True)
            uet_mask = uet_mask.cuda(non_blocking=True)
            bing_ids = bing_ids.cuda(non_blocking=True)
            bing_mask = bing_mask.cuda(non_blocking=True)
            user_ids  = user_ids.cuda(non_blocking=True)

        if args.process_uet or args.process_bing:
            user_vecs, user_vec_on = model.user_encoder(log_vecs, log_mask, uet_ids, \
            uet_mask, bing_ids, bing_mask,user_ids)
            user_vecs = user_vecs.to(torch.device("cpu")).detach().numpy()
            user_vec_on = user_vec_on.to(torch.device("cpu")).detach().numpy()
        else:
            user_vecs = model.user_encoder(log_vecs, log_mask, uet_ids, \
            uet_mask, bing_ids, bing_mask, user_ids).to(torch.device("cpu")).detach().numpy()

        for index, user_vec, news_vec, bias, label, his_len in zip(
                range(len(labels)), user_vecs, news_vecs, news_bias, labels, his_lens):
                
            if label.mean() == 0 or label.mean() == 1:
                continue

            score = np.dot(
                news_vec, user_vec
            )

            if args.process_uet or args.process_bing:
                score_on = np.dot(
                news_vec, user_vec_on[index]
            )

            if args.debias:
                score_add_bias = score + bias

            
            auc = roc_auc_score(label, score)
            mrr = mrr_score(label, score)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)
            ctr1 = ctr_score(label, score, k=1)
            ctr3 = ctr_score(label, score, k=3)
            ctr5 = ctr_score(label, score, k=5)
            ctr10 = ctr_score(label, score, k=10)


            if args.debias:
                auc_add_bias = roc_auc_score(label, score_add_bias)
                mrr_add_bias = mrr_score(label, score_add_bias)
                ndcg5_add_bias = ndcg_score(label, score_add_bias, k=5)
                ndcg10_add_bias = ndcg_score(label, score_add_bias, k=10)
                ctr1_add_bias = ctr_score(label, score_add_bias, k=1)
                ctr3_add_bias = ctr_score(label, score_add_bias, k=3)
                ctr5_add_bias = ctr_score(label, score_add_bias, k=5)
                ctr10_add_bias = ctr_score(label, score_add_bias, k=10)

            if args.process_uet or args.process_bing:
                auc_on = roc_auc_score(label, score_on)
                mrr_on = mrr_score(label, score_on)
                ndcg5_on = ndcg_score(label, score_on, k=5)
                ndcg10_on = ndcg_score(label, score_on, k=10)
                ctr1_on = ctr_score(label, score_on, k=1)
                ctr3_on = ctr_score(label, score_on, k=3)
                ctr5_on = ctr_score(label, score_on, k=5)
                ctr10_on = ctr_score(label, score_on, k=10)

            AUC[0].append(auc)
            MRR[0].append(mrr)
            nDCG5[0].append(ndcg5)
            nDCG10[0].append(ndcg10)
            CTR1[0].append(ctr1)
            CTR3[0].append(ctr3)
            CTR5[0].append(ctr5)
            CTR10[0].append(ctr10)

            if args.debias:
                AUC_add_bias[0].append(auc_add_bias)
                MRR_add_bias[0].append(mrr_add_bias)
                nDCG5_add_bias[0].append(ndcg5_add_bias)
                nDCG10_add_bias[0].append(ndcg10_add_bias)
                CTR1_add_bias[0].append(ctr1_add_bias)
                CTR3_add_bias[0].append(ctr3_add_bias)
                CTR5_add_bias[0].append(ctr5_add_bias)
                CTR10_add_bias[0].append(ctr10_add_bias)

            if args.process_uet or args.process_bing:
                AUC_on[0].append(auc_on)
                MRR_on[0].append(mrr_on)
                nDCG5_on[0].append(ndcg5_on)
                nDCG10_on[0].append(ndcg10_on)
                CTR1_on[0].append(ctr1_on)
                CTR3_on[0].append(ctr3_on)
                CTR5_on[0].append(ctr5_on)
                CTR10_on[0].append(ctr10_on)

            if his_len <= 5:
                AUC[1].append(auc)
                MRR[1].append(mrr)
                nDCG5[1].append(ndcg5)
                nDCG10[1].append(ndcg10)
                CTR1[1].append(ctr1)
                CTR3[1].append(ctr3)
                CTR5[1].append(ctr5)
                CTR10[1].append(ctr10)

                if args.debias:
                    AUC_add_bias[1].append(auc_add_bias)
                    MRR_add_bias[1].append(mrr_add_bias)
                    nDCG5_add_bias[1].append(ndcg5_add_bias)
                    nDCG10_add_bias[1].append(ndcg10_add_bias)
                    CTR1_add_bias[1].append(ctr1_add_bias)
                    CTR3_add_bias[1].append(ctr3_add_bias)
                    CTR5_add_bias[1].append(ctr5_add_bias)
                    CTR10_add_bias[1].append(ctr10_add_bias)

                if args.process_uet or args.process_bing:
                    AUC_on[1].append(auc_on)
                    MRR_on[1].append(mrr_on)
                    nDCG5_on[1].append(ndcg5_on)
                    nDCG10_on[1].append(ndcg10_on)
                    CTR1_on[1].append(ctr1_on)
                    CTR3_on[1].append(ctr3_on)
                    CTR5_on[1].append(ctr5_on)
                    CTR10_on[1].append(ctr10_on)

        if cnt % args.log_steps == 0:
            for i in range(2):
                print_metrics(hvd_rank, cnt * args.batch_size, get_mean([AUC[i], MRR[i], nDCG5[i], \
                nDCG10[i], CTR1[i], CTR3[i], CTR5[i], CTR10[i]]))
                if args.debias:
                    print_metrics(hvd_local_rank, cnt * args.batch_size, get_mean([AUC_add_bias[i], MRR_add_bias[i], nDCG5_add_bias[i], \
                    nDCG10_add_bias[i], CTR1_add_bias[i], CTR3_add_bias[i], CTR5_add_bias[i], CTR10_add_bias[i]]))
                if args.process_uet or args.process_bing:
                    print_metrics(hvd_local_rank, cnt * args.batch_size, get_mean([AUC_on[i], MRR_on[i], nDCG5_on[i], \
                    nDCG10_on[i], CTR1_on[i], CTR3_on[i], CTR5_on[i], CTR10_on[i]]))

    
    # stop scoring
    dataloader.join()

    for i in range(2):
        print_metrics(hvd_rank, cnt * args.batch_size, get_mean([AUC[i], MRR[i], nDCG5[i], \
                nDCG10[i], CTR1[i], CTR3[i], CTR5[i], CTR10[i]]))
        if args.debias:
            print_metrics(hvd_local_rank, cnt * args.batch_size, get_mean([AUC_add_bias[i], MRR_add_bias[i], nDCG5_add_bias[i], \
            nDCG10_add_bias[i], CTR1_add_bias[i], CTR3_add_bias[i], CTR5_add_bias[i], CTR10_add_bias[i]]))

if __name__ == "__main__":
    utils.setuplogger()
    args = parse_args()
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    if 'train' in args.mode:
        train(args)
    if 'test' in args.mode:
        test(args)
