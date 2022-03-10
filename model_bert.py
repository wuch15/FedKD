import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import math


class AdditiveAttention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news vectors
    Arg: 
        d_h: the last dimension of input
    '''
    def __init__(self, d_h, hidden_size=200):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, candidate_vector_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        #       [bz, 20, seq_len, 20] x [bz, 20, 20, seq_len] -> [bz, 20, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        #       [bz, 20, seq_len, seq_len] x [bz, 20, seq_len, 20] -> [bz, 20, seq_len, 20]
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, enable_gpu):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 300
        self.n_heads = n_heads  # 20
        self.d_k = d_k  # 20
        self.d_v = d_v  # 20
        self.enable_gpu = enable_gpu

        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 300, 400

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K, V, mask=None):
        #       Q, K, V: [bz, seq_len, 300] -> W -> [bz, seq_len, 400]-> q_s: [bz, 20, seq_len, 20]
        batch_size, seq_len, _ = Q.shape

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads,
                               self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len) #  [bz, seq_len, seq_len]
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [bz, 20, seq_len, seq_len]

        context, attn = ScaledDotProductAttention(self.d_k)(
            q_s, k_s, v_s, mask)  # [bz, 20, seq_len, 20]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v)  # [bz, seq_len, 400]
        #         output = self.fc(context)
        return context  #self.layer_norm(output + residual)


class WeightedLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(WeightedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight_softmax =  nn.Softmax(dim=-1)(self.weight)
        return F.linear(input, weight_softmax)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
class PositionEmbedding(nn.Module):

    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError('Unknown mode: %s' % self.mode)

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(
            self.num_embeddings, self.embedding_dim, self.mode,
        )
class TextEncoder(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 word_embedding_dim,
                 num_attention_heads,
                 query_vector_dim,
                 dropout_rate,
                 enable_gpu,layers):
        super(TextEncoder, self).__init__()
        #self.word_embedding = word_embedding
        self.bert_model = bert_model
        self.dropout_rate = dropout_rate
        self.nlayer=layers
        #self.layernorm=torch.nn.LayerNorm(num_attention_heads*20)
        #self.multihead_attention = MultiHeadAttention(word_embedding_dim,
        #                                              num_attention_heads, 20,
        #                                              20, enable_gpu)
        self.additive_attention = AdditiveAttention(768,query_vector_dim)

    def forward(self, text, mask=None):
        """
        Args:
            text: Tensor(batch_size) * num_words_text * embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_words_text
        batch_size, num_words = text.shape
        num_words = num_words // 3
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_type = torch.narrow(text, 1, num_words, num_words)
        text_attmask = torch.narrow(text, 1, num_words*2, num_words)
        temp=self.bert_model(text_ids, text_type, text_attmask)
        word_emb = temp[3][self.nlayer]
        word_emb2 = temp[3]
        word_att=temp[4]
        text_vector = F.dropout(word_emb,
                                p=self.dropout_rate,
                                training=self.training)
        # batch_size, num_words_text, word_embedding_dim
        #multihead_text_vector = self.multihead_attention(
        #    text_vector, text_vector, text_vector, mask)
        #multihead_text_vector = F.dropout(text_vector,
        #                                  p=self.dropout_rate,
        #                                  training=self.training)
        # batch_size, word_embedding_dim
        text_vector = self.additive_attention(text_vector, mask)
        return text_vector,word_emb,torch.stack(word_emb2,dim=1).cuda(),torch.stack(word_att,dim=1).cuda()


class ElementEncoder(torch.nn.Module):
    def __init__(self, num_elements, embedding_dim, enable_gpu=True):
        super(ElementEncoder, self).__init__()
        self.enable_gpu = enable_gpu
        self.embedding = nn.Embedding(num_elements,
                                      embedding_dim,
                                      padding_idx=0)

    def forward(self, element):
        # batch_size, embedding_dim
        element_vector = self.embedding(
            (element.cuda() if self.enable_gpu else element).long())
        return element_vector


class NewsEncoder(torch.nn.Module):
    def __init__(self, args, bert_model, category_dict_size,
                 domain_dict_size, subcategory_dict_size,layers):
        super(NewsEncoder, self).__init__()
        self.args = args
        self.attributes2length = {
            'title': args.num_words_title * 3,
            'abstract': args.num_words_abstract * 3,
            'body': args.num_words_body * 3,
            'category': 1,
            'domain': 1,
            'subcategory': 1
        }
        for key in list(self.attributes2length.keys()):
            if key not in args.news_attributes:
                self.attributes2length[key] = 0

        self.attributes2start = {
            key: sum(
                list(self.attributes2length.values())
                [:list(self.attributes2length.keys()).index(key)])
            for key in self.attributes2length.keys()
        }

        assert len(args.news_attributes) > 0
        text_encoders_candidates = ['title', 'body']

        self.text_encoders = nn.ModuleDict({
            'title':
            TextEncoder(bert_model,
                        args.word_embedding_dim,
                        args.num_attention_heads, args.news_query_vector_dim,
                        args.drop_rate, args.enable_gpu,layers)
        })
    
        self.newsname=[name for name in set(args.news_attributes) & set(text_encoders_candidates)]
         
        name2num = {
            "category": category_dict_size + 1,
            "domain": domain_dict_size + 1,
            "subcategory": subcategory_dict_size + 1
        }
        element_encoders_candidates = ['category', 'domain', 'subcategory']
        self.element_encoders = nn.ModuleDict({
            name: ElementEncoder(name2num[name], 
                                args.num_attention_heads * 20,
                                 args.enable_gpu)
            for name in (set(args.news_attributes)
                         & set(element_encoders_candidates))
        })

        self.reduce_dim_linear = nn.Linear(768,args.news_dim)
        

    def forward(self, news):
        """
        Args:
        Returns:
            (shape) batch_size, news_dim
        """
        text_vectors = self.text_encoders['title'](torch.narrow(news, 1, self.attributes2start['title'],self.attributes2length['title']))

        final_news_vector = text_vectors[0]

        # batch_size, news_dim
        final_news_vector = self.reduce_dim_linear(final_news_vector)
        return final_news_vector,text_vectors[1],text_vectors[2],text_vectors[3]


class UETBingEncoder(torch.nn.Module):
    def __init__(self, args):
        super(UETBingEncoder, self).__init__()
        self.args = args
        self.behavior_encoder = TextEncoder(
                            args.word_embedding_dim,
                            args.num_attention_heads,
                            args.news_query_vector_dim,
                            args.drop_rate,
                            args.enable_gpu)
        self.reduce_dim_linear = nn.Linear(args.num_attention_heads * 20,
                                           args.news_dim)
        if args.use_pretrain_news_encoder:
            self.behavior_encoder.load_state_dict(
                    torch.load(os.path.join(args.pretrain_news_encoder_path, 
                    'behavior_encoder.pkl'))
                )
            self.reduce_dim_linear.load_state_dict(
                torch.load(os.path.join(args.pretrain_news_encoder_path, 
                'reduce_dim_linear.pkl'))
            )

    def forward(self, behavior_ids):
        behavior_vector = self.behavior_encoder(behavior_ids)
        behavior_vector = self.reduce_dim_linear(behavior_vector)
        return behavior_vector


class UserEncoder(torch.nn.Module):
    def __init__(self, args, uet_encoder=None, uet_reduce_linear=None,
                bing_encoder=None, bing_reduce_linear=None):
        super(UserEncoder, self).__init__()
        self.args = args
        #self.userembedding = nn.Embedding(900000,32,padding_idx=0)
        #self.dense=
        #self.layernorm=torch.nn.LayerNorm(args.news_dim)
        #self.posemb=PositionEmbedding(num_embeddings=50, embedding_dim=args.news_dim, mode=PositionEmbedding.MODE_ADD)
        #self.news_multihead_attention = MultiHeadAttention(args.news_dim,
        #                                              20, 20,
        #                                              20, self.args.enable_gpu)
        #self.dense = nn.Linear(args.news_dim+32,args.news_dim)
        self.news_additive_attention = AdditiveAttention(
            args.news_dim, args.user_query_vector_dim)
        if args.use_padded_news_embedding:
            # self.news_padded_news_embedding = nn.Embedding(1, args.news_dim)
            self.pad_doc = nn.Parameter(torch.empty(1, args.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)
        else:
            # self.news_padded_news_embedding = None
            self.pad_doc = None

        if args.process_uet:
            if args.title_share_encoder:
                self.uet_encoder = nn.Sequential(
                            uet_encoder, uet_reduce_linear)
            else:
                self.uet_encoder = UETBingEncoder(args)
            self.uet_additive_attention = AdditiveAttention(
            args.news_dim, args.user_query_vector_dim)

        if args.process_bing:
            if args.title_share_encoder:
                self.bing_encoder =  nn.Sequential(
                            bing_encoder, bing_reduce_linear)
            else:
                self.bing_encoder = UETBingEncoder(args)
            self.bing_additive_attention = AdditiveAttention(
            args.news_dim, args.user_query_vector_dim)
        
        if args.process_uet or args.process_bing:
            if args.uet_agg_method == 'attention':
                self.user_behavior_att = AdditiveAttention(
                    args.news_dim, args.user_query_vector_dim)
            elif args.uet_agg_method == 'weighted-sum':
                self.behavior_linear = WeightedLinear(
                    3 if args.process_bing and args.process_uet else 2, 1)

    def _process_news(self, vec, mask, pad_doc,
                    additive_attention, use_mask=False, 
                    use_padded_embedding=False):
        assert not (use_padded_embedding and use_mask), 'Conflicting config'
        if use_padded_embedding:
            # batch_size, maxlen, dim
            batch_size = vec.shape[0]
            padding_doc = pad_doc.expand(batch_size, self.args.news_dim).unsqueeze(1).expand( \
                                         batch_size, self.args.user_log_length , self.args.news_dim)
            # batch_size, maxlen, dim
            vec = vec * mask.unsqueeze(2).expand(-1, -1, self.args.news_dim) + padding_doc * (1 - mask.unsqueeze(2).expand(-1, -1, self.args.news_dim))
        # batch_size, news_dim
        vec = additive_attention(vec,
                                 mask if use_mask else None)
        return vec
    
    def _process_uet_bing(self, vec, mask, additive_attention):
        batch_size = vec.size(0)
        vec = additive_attention(vec, mask)
        if self.training:
            mask_v = torch.empty(batch_size).bernoulli_(self.args.mask_uet_bing_rate)
            if self.args.enable_gpu:
                mask_v = mask_v.cuda()
            vec = vec * mask_v.unsqueeze(1).expand_as(vec)
        return vec

    
    def forward(self, log_vec, log_mask, uet_ids=None, uet_mask=None, bing_ids=None, bing_mask=None, user_ids=None):
        """
        Returns:
            (shape) batch_size,  news_dim
        """
        # batch_size, news_dim
        #log_vec = self.posemb(log_vec)
        #log_vec = self.news_multihead_attention(log_vec,log_vec,log_vec,log_mask)
        #user_vec = self.userembedding(user_ids).view(-1, self.args.news_dim)
        log_vec = self._process_news(log_vec, log_mask, self.pad_doc,
                                     self.news_additive_attention, self.args.user_log_mask,
                                     self.args.use_padded_news_embedding)
        #print(user_vec.size())
        #user_log_vecs = [dense(torch.cat((log_vec, user_vec), dim=-1))]
        #user_log_vecs = [torch.sum(
        #        torch.stack([log_vec,user_vec], dim=1),
        #        dim=1
        #     )] 
        user_log_vecs = [log_vec]

        if self.args.process_uet:
            batch_size, user_uet_length, num_words_uet = uet_ids.shape
            uet_ids = uet_ids.view(-1, num_words_uet)
            uet_vec = self.uet_encoder(uet_ids).reshape(batch_size, user_uet_length, -1)
            uet_vec = self._process_uet_bing(uet_vec, uet_mask, self.uet_additive_attention)
            user_log_vecs.append(uet_vec)
        
        if self.args.process_bing:
            batch_size, user_bing_length, num_words_bing = bing_ids.shape
            bing_ids = bing_ids.view(-1, num_words_bing)
            bing_vec = self.bing_encoder(bing_ids).reshape(batch_size, user_bing_length, -1)
            bing_vec = self._process_uet_bing(bing_vec, bing_mask, self.bing_additive_attention)
            user_log_vecs.append(bing_vec)

        if len(user_log_vecs) == 1:
            return user_log_vecs[0]
        else:
            if self.args.uet_agg_method == 'attention':
                return self.user_behavior_att(torch.stack(user_log_vecs, dim=1)), user_log_vecs[0]
            if self.args.uet_agg_method == 'sum':
                return torch.sum(torch.stack(user_log_vecs, dim=1), dim=1), user_log_vecs[0]
            if self.args.uet_agg_method == 'weighted-sum':
                return (
                    self.behavior_linear(torch.stack(user_log_vecs, dim=1).\
                    transpose(-1, -2)).squeeze(dim=-1), \
                    user_log_vecs[0]
                    )
                


class ModelBert(torch.nn.Module):
    """
    UniUM network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self,
                 args,
                 bert_model_tea,bert_model_stu,
                 category_dict_size=0,
                 domain_dict_size=0,
                 subcategory_dict_size=0):
        super(ModelBert, self).__init__()
        self.args = args

        self.teacher_bert_models = NewsEncoder(args,bert_model_tea,category_dict_size, domain_dict_size,subcategory_dict_size,12)
        self.student_bert_models = NewsEncoder(args,bert_model_stu,category_dict_size, domain_dict_size,subcategory_dict_size,4)
        self.mdenses = [nn.Linear(768, 768) for layers in range(4)]
        for dense in self.mdenses:
            nn.init.xavier_uniform_(dense.weight, gain=1)
        self.user_encoder = UserEncoder(args)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion2 = nn.KLDivLoss()
        self.criterion3 = nn.MSELoss()

    def forward(self,
                input_ids,
                log_ids,
                log_mask,
                targets=None,
                uet_ids=None,
                uet_mask=None,
                bing_ids=None,
                bing_mask=None,
                user_ids=None,
                random_news=None,
                targets2=None,
                compute_loss=True):
        """
        Returns:
          click_probability: batch_size, 1 + K
        """
        # input_ids: batch, history, num_words
        ids_length = input_ids.size(2)
        input_ids = input_ids.view(-1, ids_length)
        news_vec, news_wordvec, news_wordvec2, news_wordatt= self.teacher_bert_models(input_ids)
        news_vec = news_vec.view(-1, 1 + self.args.npratio, self.args.news_dim)
        
        news_vec2, news2_wordvec, news2_wordvec2, news2_wordatt = self.student_bert_models(input_ids)
        news_vec2 = news_vec2.view(-1, 1 + self.args.npratio, self.args.news_dim)

        # batch_size, news_dim
        log_ids = log_ids.view(-1, ids_length)
        log_vec, log_wordvec, log_wordvec2, log_wordatt = self.teacher_bert_models(log_ids)
        log_vec = log_vec.view(-1, self.args.user_log_length,self.args.news_dim)
        log_vec2, log2_wordvec, log2_wordvec2, log2_wordatt  = self.student_bert_models(log_ids)
        log_vec2 = log_vec2.view(-1, self.args.user_log_length,self.args.news_dim)

        user_vector = self.user_encoder(log_vec, log_mask, uet_ids, uet_mask, bing_ids, bing_mask, user_ids)
        user_vector2 = self.user_encoder(log_vec2, log_mask, uet_ids, uet_mask, bing_ids, bing_mask, user_ids)
        
        # batch_size, 2
        score = torch.bmm(news_vec, user_vector.unsqueeze(-1)).squeeze(dim=-1)  
        score2 = torch.bmm(news_vec2, user_vector2.unsqueeze(-1)).squeeze(dim=-1)    
        
        loss_mse=0.
        for layer in range(news2_wordvec2.size()[1]-1):
            #print(layer)
            layerratio=12//4
            loss_mse+=self.criterion3(news2_wordvec2[:,-layer-1],news_wordvec2[:,-layer*layerratio-1])+self.criterion3(log2_wordvec2[:,-layer-1],log_wordvec2[:,-layer*layerratio-1])
            loss_mse+=self.criterion3(news_wordatt[:,-layer*layerratio-1],news2_wordatt[:,-layer-1])+self.criterion3(log_wordatt[:,-layer*layerratio-1],log2_wordatt[:,-layer-1])
        
        if compute_loss:
            outputs_S1 = F.log_softmax(score, dim=1)
            outputs_S2 = F.log_softmax(score2, dim=1)
            outputs_T1 = F.softmax(score, dim=1)
            outputs_T2= F.softmax(score2, dim=1)

            loss1 = self.criterion(score, targets) 
            loss2 = self.criterion(score2, targets) 
            loss3 = self.criterion2(outputs_S1, outputs_T2)/(loss1+loss2)
            loss4 = self.criterion2(outputs_S2, outputs_T1)/(loss1+loss2)
            
            
            return loss1+loss3+loss_mse/(loss1+loss2)+loss2+loss4+loss_mse/(loss1+loss2), score 
        
        
            
        
        else:
            return score,score2


if __name__ == "__main__":
    from parameters import parse_args

    from tnlrv3.modeling import TuringNLRv3ForSequenceClassification
    from tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
    from tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer

    from news_word_embedding import infer_news

    args = parse_args()

    MODEL_CLASSES = {
        'tnlrv3': (TuringNLRv3Config, TuringNLRv3ForSequenceClassification, TuringNLRv3Tokenizer),
    }
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    bert_model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    bert_model=bert_model.load_state_dict(torch.load('../bert_encoder.pkl'))

    args.news_attributes = ['title']
    args.debias=False
    args.process_uet = False
    args.process_bing = False
    args.user_log_mask=False
    args.padded_news_different_word_index = False
    #args.news_attributes = ['title', 'body']

    args.use_pretrain_news_encoder = False
    args.title_share_encoder = False
    args.debias = False
    args.uet_agg_method = 'weighted-sum'

    args.pretrain_news_encoder_path = "./model_all/pretrain_textencoder/"

    #word_dict = torch.load(os.path.join(args.pretrain_news_encoder_path, 'word_dict.pkl'))

    #embedding_matrix = np.random.uniform(size=(len(word_dict)+1, args.word_embedding_dim))
    model = ModelBert(args, bert_model, 10, 10, 10)
    model.cuda()
    length = args.num_words_title * 3
    input_ids = torch.ones((64, 2, length)).cuda().long()
    log_ids = torch.ones((64, 50, length)).cuda().long()
    log_mask = torch.rand((64, 50)).cuda().float()
    targets = torch.rand((64, )).cuda().long()
    uet_ids = torch.rand(64, 30, 16).cuda().long()
    uet_mask = torch.rand(64, 30).cuda().float()
    bing_ids = torch.LongTensor([]).cuda()
    bing_mask = torch.FloatTensor([]).cuda()
    print(model(input_ids, log_ids, log_mask, targets, uet_ids, uet_mask, bing_ids, bing_mask))
