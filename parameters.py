import argparse
import utils
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        default="train_test",
                        choices=['train', 'test', 'train_test'])
    parser.add_argument(
        "--root_data_dir",
        type=str,
        default=
        "adl://udc-c09.azuredatalakestore.net:443/local/RnR/users/v-jinyi/MSNPipeline/MultiLingle",
    )
    parser.add_argument("--market",
                        type=str,
                        default='pt-pt',
                        choices=[
                            'MIND', 'de-at', 'de-de', 'en-au', 'en-ca', 'en-gb',
                            'en-in', 'en-us', 'es-es', 'es-mx', 'fr-ca',
                            'fr-fr', 'it-it', 'pt-br', 'pt-pt'
                        ])
    parser.add_argument(
        "--train_date",
        type=str,
        default='2020-06-30',
    )
    parser.add_argument(
        "--test_date",
        type=str,
        default='2020-07-31',
    )
    parser.add_argument("--filename_pat", type=str, default="behaviors_*.tsv")
    parser.add_argument("--model_dir", type=str, default='./model')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--npratio", type=int, default=1)
    parser.add_argument("--enable_gpu", type=utils.str2bool, default=True)
    parser.add_argument("--enable_hvd", type=utils.str2bool, default=True)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--filter_num", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1000)
    parser.add_argument("--tmin", type=int, default=0.95)
    parser.add_argument("--tmax", type=int, default=0.98)

    # model training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--news_attributes",
        type=str,
        nargs='+',
        default=['title'],
        choices=['title', 'abstract', 'body', 'category', 'domain', 'subcategory'])
    parser.add_argument("--process_uet", type=utils.str2bool, default=False)
    parser.add_argument("--process_bing", type=utils.str2bool, default=False)
       
    parser.add_argument("--num_words_title", type=int, default=24)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=50)
    parser.add_argument("--num_words_uet", type=int, default=16)
    parser.add_argument("--num_words_bing", type=int, default=26)
    parser.add_argument(
        "--user_log_length",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--user_bing_length",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--user_uet_length",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--word_embedding_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--glove_embedding_path",
        type=str,
        default='/feeds-nfs/data/v-jinyi/glove.840B.300d.txt',
    )
    parser.add_argument(
        "--muse_embedding_dir",
        type=str,
        default='/feeds-nfs/data/v-leiyu/MUSE',
    )
    parser.add_argument("--embedding_source",
                        type=str,
                        default='random',
                        choices=['glove', 'muse', 'random'])
    parser.add_argument("--freeze_embedding",
                        type=utils.str2bool,
                        default=False)
    parser.add_argument("--use_padded_news_embedding",
                        type=utils.str2bool,
                        default=False)
    parser.add_argument("--padded_news_different_word_index",
                        type=utils.str2bool,
                        default=False)
    parser.add_argument("--debias", type=utils.str2bool, default=True)
    parser.add_argument(
        "--news_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--news_query_vector_dim",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--user_query_vector_dim",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=20,
    )
    parser.add_argument("--user_log_mask", type=utils.str2bool, default=True)
    parser.add_argument("--mask_uet_bing_rate", type=float, default=0.5)
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_steps_per_epoch", type=int, default=1000000)

    parser.add_argument(
        "--load_ckpt_name",
        type=str,
        default=None,
        help="choose which ckpt to load and test"
    )
    # share
    parser.add_argument("--title_share_encoder", type=utils.str2bool, default=False)

    # pretrain
    parser.add_argument("--use_pretrain_news_encoder", type=utils.str2bool, default=False)
    parser.add_argument("--pretrain_news_encoder_path", type=str, default=".")
    
    # uet add method
    parser.add_argument(
        "--uet_agg_method", 
        type=str, 
        default='attention', 
        choices=['sum', 'attention', 'weighted-sum'])

    # turing
    parser.add_argument("--model_type", default="tnlrv3", type=str)
    parser.add_argument("--apply_turing", type=utils.str2bool, default=True)
    parser.add_argument("--do_lower_case", type=utils.str2bool, default=True)
    parser.add_argument("--model_name_or_path", default="../Turing/unilm2-base-uncased.bin", type=str,
                        help="Path to pre-trained model or shortcut name. ")
    parser.add_argument("--config_name", default="../Turing/unilm2-base-uncased-config.json", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="../Turing/unilm2-base-uncased-vocab.txt", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    args = parser.parse_args()

    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
