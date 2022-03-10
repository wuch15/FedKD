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

class TextEncoder(torch.nn.Module):
    def __init__(self,
                 word_embedding,
                 word_embedding_dim,
                 num_attention_heads,
                 query_vector_dim,
                 dropout_rate,
                 enable_gpu=True):
        super(TextEncoder, self).__init__()
        self.word_embedding = word_embedding
        self.dropout_rate = dropout_rate
        self.multihead_attention = MultiHeadAttention(word_embedding_dim,
                                                      num_attention_heads, 20,
                                                      20, enable_gpu)
        self.additive_attention = AdditiveAttention(num_attention_heads * 20,
                                                    query_vector_dim)

    def forward(self, text, mask=None):
        """
        Args:
            text: Tensor(batch_size) * num_words_text
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_words_text, word_embedding_dim
        text_vector = F.dropout(self.word_embedding(text.long()),
                                p=self.dropout_rate,
                                training=self.training)
        # batch_size, num_words_text, word_embedding_dim
        multihead_text_vector = self.multihead_attention(
            text_vector, text_vector, text_vector, mask)
        multihead_text_vector = F.dropout(multihead_text_vector,
                                          p=self.dropout_rate,
                                          training=self.training)
        # batch_size, word_embedding_dim
        text_vector = self.additive_attention(multihead_text_vector, mask)
        return text_vector


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
    def __init__(self, args, word_embedding, category_dict_size,
                 domain_dict_size, subcategory_dict_size):
        super(NewsEncoder, self).__init__()
        self.args = args
        self.attributes2length = {
            'title': args.num_words_title,
            'abstract': args.num_words_abstract,
            'body': args.num_words_body,
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

        self.word_embedding = word_embedding
        assert len(args.news_attributes) > 0
        text_encoders_candidates = ['title', 'abstract']
        self.text_encoders = nn.ModuleDict({
            name:
            TextEncoder(self.word_embedding, args.word_embedding_dim,
                        args.num_attention_heads, args.news_query_vector_dim,
                        args.drop_rate, args.enable_gpu)
            for name in (set(args.news_attributes)
                         & set(text_encoders_candidates))
        })

        if args.use_pretrain_news_encoder:
            for name in self.text_encoders:
                self.text_encoders[name].load_state_dict(
                    torch.load(os.path.join(args.pretrain_news_encoder_path, 
                    'behavior_encoder.pkl'))
                )

        if 'body' in args.news_attributes:
            assert args.num_words_body % 4 == 0
            self.num_words_body_segment = args.num_words_body // 4
            self.body_encoder =  TextEncoder(self.word_embedding, args.word_embedding_dim,
                            args.num_attention_heads,
                            args.news_query_vector_dim, args.drop_rate,
                            args.enable_gpu)

            if args.use_pretrain_news_encoder:
                self.body_encoder.load_state_dict(
                    torch.load(os.path.join(args.pretrain_news_encoder_path, 
                    'behavior_encoder.pkl'))
                )

            
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
        if len(args.news_attributes) > 1:
            self.final_attention = AdditiveAttention(
                args.num_attention_heads * 20, args.news_query_vector_dim)

        self.reduce_dim_linear = nn.Linear(args.num_attention_heads * 20,
                                           args.news_dim)

        if args.use_pretrain_news_encoder:
            self.reduce_dim_linear.load_state_dict(
                torch.load(os.path.join(args.pretrain_news_encoder_path, 
                'reduce_dim_linear.pkl'))
            )

    def forward(self, news):
        """
        Args:
        Returns:
            (shape) batch_size, news_dim
        """
        text_vectors = [
            encoder(
                torch.narrow(news, 1, self.attributes2start[name],
                             self.attributes2length[name]))
            for name, encoder in self.text_encoders.items()
        ]
        if 'body' in self.args.news_attributes:
            body = torch.narrow(news, 1, self.attributes2start['body'],
                                self.args.num_words_body)
            body = body.reshape(-1, self.num_words_body_segment)
            body_vector = self.body_encoder(body)
            body_vector = body_vector.view(-1, 4, body_vector.size(-1))
            body_vector = torch.mean(body_vector, dim=1)

            text_vectors.append(body_vector)

        element_vectors = [
            encoder(
                torch.narrow(news, 1, self.attributes2start[name],
                             self.attributes2length[name]).squeeze(dim=1))
            for name, encoder in self.element_encoders.items()
        ]

        all_vectors = text_vectors + element_vectors

        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:
            final_news_vector = self.final_attention(
                torch.stack(all_vectors, dim=1))

            # final_news_vector = torch.mean(
            #     torch.stack(all_vectors, dim=1),
            #     dim=1
            # )
        # batch_size, news_dim
        #final_news_vector = self.reduce_dim_linear(final_news_vector)
        return final_news_vector


class UETBingEncoder(torch.nn.Module):
    def __init__(self, args, word_embedding):
        super(UETBingEncoder, self).__init__()
        self.args = args
        self.behavior_encoder = TextEncoder(
                            word_embedding,
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
    def __init__(self, args, word_embedding, uet_encoder=None, uet_reduce_linear=None,
                bing_encoder=None, bing_reduce_linear=None):
        super(UserEncoder, self).__init__()
        self.args = args

        self.news_multihead_attention = MultiHeadAttention(args.num_attention_heads * 20,
                                                      args.num_attention_heads, 20,
                                                      20, args.enable_gpu)


        self.news_additive_attention = AdditiveAttention(
            args.num_attention_heads * 20, args.user_query_vector_dim)
        if args.use_padded_news_embedding:
            # self.news_padded_news_embedding = nn.Embedding(1, args.num_attention_heads * 20)
            self.pad_doc = nn.Parameter(torch.empty(1, args.num_attention_heads * 20).uniform_(-1, 1)).type(torch.FloatTensor)
        else:
            # self.news_padded_news_embedding = None
            self.pad_doc = None

        if args.process_uet:
            if args.title_share_encoder:
                self.uet_encoder = nn.Sequential(
                            uet_encoder, uet_reduce_linear)
            else:
                self.uet_encoder = UETBingEncoder(args, word_embedding)
            self.uet_additive_attention = AdditiveAttention(
            args.news_dim, args.user_query_vector_dim)

        if args.process_bing:
            if args.title_share_encoder:
                self.bing_encoder =  nn.Sequential(
                            bing_encoder, bing_reduce_linear)
            else:
                self.bing_encoder = UETBingEncoder(args, word_embedding)
            self.bing_additive_attention = AdditiveAttention(
            args.news_dim, args.user_query_vector_dim)
        
        if args.process_uet or args.process_bing:
            if args.uet_agg_method == 'attention':
                self.user_behavior_att = AdditiveAttention(
                    args.news_dim, args.user_query_vector_dim)
            elif args.uet_agg_method == 'weighted-sum':
                self.behavior_linear = WeightedLinear(
                    3 if args.process_bing and args.process_uet else 2, 1)
    
    def get_user_news_scoring(self, log_vec, log_mask):
        log_vec = self._process_news(log_vec, log_mask, self.pad_doc,
                                     self.news_multihead_attention,
                                     self.news_additive_attention, self.args.user_log_mask,
                                     self.args.use_padded_news_embedding)

        return log_vec

    def get_user_uet_scoring(self, uet_ids, uet_mask):
        batch_size, user_uet_length, num_words_uet = uet_ids.shape
        uet_ids = uet_ids.view(-1, num_words_uet)
        uet_vec = self.uet_encoder(uet_ids).reshape(batch_size, user_uet_length, -1)
        uet_vec = self._process_uet_bing(uet_vec, uet_mask, self.uet_additive_attention)

        return uet_vec
    
    def get_user_bing_scoring(self, bing_ids, bing_mask):
        batch_size, user_bing_length, num_words_bing = bing_ids.shape
        bing_ids = bing_ids.view(-1, num_words_bing)
        bing_vec = self.bing_encoder(bing_ids).reshape(batch_size, user_bing_length, -1)
        bing_vec = self._process_uet_bing(bing_vec, bing_mask, self.bing_additive_attention)

        return bing_vec
        
    def _process_news(self, vec, mask, pad_doc,
                    multihead_attention, additive_attention, use_mask=False, 
                    use_padded_embedding=False):
        assert not (use_padded_embedding and use_mask), 'Conflicting config'
        if use_padded_embedding:
            # batch_size, maxlen, dim
            batch_size = vec.shape[0]
            padding_doc = pad_doc.expand(batch_size, self.args.num_attention_heads * 20).unsqueeze(1).expand( \
                                         batch_size, self.args.user_log_length , self.args.num_attention_heads * 20)
            # batch_size, maxlen, dim
            vec = vec * mask.unsqueeze(2).expand(-1, -1, self.args.num_attention_heads * 20) + padding_doc * (1 - mask.unsqueeze(2).expand(-1, -1, self.args.num_attention_heads * 20))
        # batch_size, news_dim
        vec = multihead_attention(vec, vec, vec, mask if use_mask else None)
        vec = F.dropout(vec, p=self.args.drop_rate, training=self.training)
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

    def get_user_news_scoring(self, log_vec, log_mask):
        log_vec = self._process_news(log_vec, log_mask, self.pad_doc,
                                     self.news_multihead_attention,
                                     self.news_additive_attention, self.args.user_log_mask,
                                     self.args.use_padded_news_embedding)

        return log_vec

    def get_user_uet_scoring(self, uet_ids, uet_mask):
        batch_size, user_uet_length, num_words_uet = uet_ids.shape
        uet_ids = uet_ids.view(-1, num_words_uet)
        uet_vec = self.uet_encoder(uet_ids).reshape(batch_size, user_uet_length, -1)
        uet_vec = self._process_uet_bing(uet_vec, uet_mask, self.uet_additive_attention)

        return uet_vec
    
    def get_user_bing_scoring(self, bing_ids, bing_mask):
        batch_size, user_bing_length, num_words_bing = bing_ids.shape
        bing_ids = bing_ids.view(-1, num_words_bing)
        bing_vec = self.bing_encoder(bing_ids).reshape(batch_size, user_bing_length, -1)
        bing_vec = self._process_uet_bing(bing_vec, bing_mask, self.bing_additive_attention)

        return bing_vec

    
    def forward(self, log_vec, log_mask, uet_ids=None, uet_mask=None, bing_ids=None, bing_mask=None):
        """
        Returns:
            (shape) batch_size,  news_dim
        """
        # batch_size, news_dim
        log_vec = self._process_news(log_vec, log_mask, self.pad_doc,
                                     self.news_multihead_attention,
                                     self.news_additive_attention, self.args.user_log_mask,
                                     self.args.use_padded_news_embedding)
        
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
                


class Model(torch.nn.Module):
    """
    UniUM network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self,
                 args,
                 embedding_matrix,
                 category_dict_size=0,
                 domain_dict_size=0,
                 subcategory_dict_size=0):
        super(Model, self).__init__()
        self.args = args

        pretrained_news_word_embedding = torch.from_numpy(embedding_matrix).float()

        if args.padded_news_different_word_index:
            padded_news_word_embedding = np.random.normal(
                size=(1, args.word_embedding_dim))
            padded_news_word_embedding = torch.from_numpy(
                padded_news_word_embedding).float()
            pretrained_news_word_embedding = torch.cat(
                [pretrained_news_word_embedding, padded_news_word_embedding],
                dim=0)

        word_embedding = nn.Embedding.from_pretrained(
            pretrained_news_word_embedding,
            freeze=args.freeze_embedding,
            padding_idx=0)

        if args.use_pretrain_news_encoder:
            word_embedding.load_state_dict(
                torch.load(os.path.join(args.pretrain_news_encoder_path, 'word_embedding.pkl'))
            )

        self.news_encoder = NewsEncoder(args, word_embedding,
                                        category_dict_size, 
                                        domain_dict_size,
                                        subcategory_dict_size)
        
        if args.debias:
            self.news_encoder_debias = NewsEncoder(args, word_embedding,
                                                category_dict_size,
                                                domain_dict_size,
                                                subcategory_dict_size)
            self.debias_linear = nn.Sequential(
                                    nn.Linear(args.num_attention_heads * 20, args.num_attention_heads * 20//2),
                                    nn.Tanh(),
                                    nn.Linear(args.num_attention_heads * 20//2, 1))
                                        
        if args.title_share_encoder:
            self.user_encoder =  UserEncoder(args, word_embedding, 
                                    uet_encoder=self.news_encoder.text_encoders['title'],
                                    bing_encoder=self.news_encoder.text_encoders['title'],
                                    uet_reduce_linear=self.news_encoder.reduce_dim_linear,
                                    bing_reduce_linear=self.news_encoder.reduce_dim_linear)
        else:
            self.user_encoder = UserEncoder(args, word_embedding)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self,
                input_ids,
                log_ids,
                log_mask,
                targets=None,
                uet_ids=None,
                uet_mask=None,
                bing_ids=None,
                bing_mask=None,
                compute_loss=True):
        """
        Returns:
          click_probability: batch_size, 1 + K
        """
        # input_ids: batch, history, num_words
        ids_length = input_ids.size(2)
        input_ids = input_ids.view(-1, ids_length)
        news_vec = self.news_encoder(input_ids)
        news_vec = news_vec.view(-1, 1 + self.args.npratio, self.args.num_attention_heads * 20)

        if self.args.debias:
            # 64, 64
            news_vec_debias = self.news_encoder_debias(input_ids)
            # 64, 1
            news_bias = self.debias_linear(news_vec_debias)
            # 32, 2
            news_bias = news_bias.view(-1, 1 + self.args.npratio)

        # batch_size, news_dim
        log_ids = log_ids.view(-1, ids_length)
        log_vec = self.news_encoder(log_ids)
        log_vec = log_vec.view(-1, self.args.user_log_length,
                               self.args.num_attention_heads * 20)

        if self.args.process_uet or self.args.process_bing:
            user_vector, _ = self.user_encoder(log_vec, log_mask, uet_ids, uet_mask, bing_ids, bing_mask)
        else:
            user_vector = self.user_encoder(log_vec, log_mask, uet_ids, uet_mask, bing_ids, bing_mask)

        # batch_size, 2
        score = torch.bmm(news_vec, user_vector.unsqueeze(-1)).squeeze(
            dim=-1) + (news_bias if self.args.debias else 0)
        if compute_loss:
            loss = self.criterion(score, targets)
            return loss, score
        else:
            return score


if __name__ == "__main__":
    from parameters import parse_args
    args = parse_args()
    args.news_attributes = ['title', 'abstract', 'category', 'domain', 'subcategory']
    args.debias=True
    args.process_uet = True
    args.process_bing = False
    args.user_log_mask=True
    args.padded_news_different_word_index = True
    #args.news_attributes = ['title', 'body']

    args.use_pretrain_news_encoder = True
    args.title_share_encoder = False
    args.debias = True
    args.uet_agg_method = 'weighted-sum'

    args.pretrain_news_encoder_path = "./model_all/pretrain_textencoder/"

    word_dict = torch.load(os.path.join(args.pretrain_news_encoder_path, 'word_dict.pkl'))

    embedding_matrix = np.random.uniform(size=(len(word_dict)+1, args.word_embedding_dim))
    model = Model(args, embedding_matrix, 10, 10, 10)
    model.cuda()
    length = args.num_words_title + args.num_words_abstract + args.num_words_body + 3
    input_ids = torch.ones((128, 2, length)).cuda()
    log_ids = torch.ones((128, 50, length)).cuda()
    log_mask = torch.rand((128, 50)).cuda().float()
    targets = torch.rand((128, )).cuda().long()
    uet_ids = torch.rand(128, 30, 16).cuda().long()
    uet_mask = torch.rand(128, 30).cuda().float()
    bing_ids = torch.LongTensor([]).cuda()
    bing_mask = torch.FloatTensor([]).cuda()
    print(model(input_ids, log_ids, log_mask, targets, uet_ids, uet_mask, bing_ids, bing_mask))
