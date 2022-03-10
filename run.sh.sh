# mode attr model_dir process_uet pretrain debias title_share_encoder apply_turing
./demo.sh train 'title'  ../model_all/model-t-tnlr False False False False sum True 768 > ../log_dir/log_t_tnlr_pre.txt
./demo.sh test 'title'  ../model_all/model-t-tnlr False False False False sum True 768 epoch-2.pt > ../log_dir/log_t_tnlr_pre_test.txt