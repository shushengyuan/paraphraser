import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam
# torch.optim是一个实现了多种优化算法的包，大多数通用的方法都已支持，提供了丰富的接口调用，未来更多精炼的优化算法也将整合进来。 

import sample 
from utils.batch_loader import BatchLoader
from model.parameters import Parameters
from model.paraphraser import Paraphraser

if __name__ == "__main__":
    '''
        当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
        当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
    '''
    parser = argparse.ArgumentParser(description='Paraphraser')
    '''
    argparse 是python自带的命令行参数解析包，可以用来方便地读取命令行参数,
    当你的代码需要频繁地修改参数的时候，使用这个工具可以将参数和代码分离开来,
    让你的代码更简洁，适用范围更广。
    
    通过argparser.ArgumentParser函数生成argparser对象,
    其中这个函数的description函数表示在命令行显示帮助信息的时候，这个程序的描述信息。
    '''
    parser.add_argument('--num-iterations', type=int, default=60000, metavar='NI',
                        help='num iterations (default: 60000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--model-name', default='', metavar='MN',
                        help='name of model to save (default: "")')
    parser.add_argument('--weight-decay', default=0.0, type=float, metavar='WD',
                        help='L2 regularization penalty (default: 0.0)')
    parser.add_argument('--use-quora', default=False, type=bool, metavar='quora', 
                    help='if include quora dataset (default: False)')
    parser.add_argument('--use-snli', default=False, type=bool, metavar='snli', 
                    help='if include snli dataset (default: False)')
    parser.add_argument('--use-coco', default=False, type=bool, metavar='coco', 
                    help='if include mscoco dataset (default: False)')
    parser.add_argument('--interm-sampling', default=False, type=bool, metavar='IS', 
                    help='if sample while training (default: False)')
    # 以上是通过对象的add_argument函数来增加参数。
    args = parser.parse_args()
    # 通过argpaser对象的parser_args函数来获取所有参数args
    
    datasets = set()
    if args.use_quora is True:
        datasets.add('quora')
    if args.use_snli is True:
        datasets.add('snli')
    if args.use_coco is True:
        datasets.add('mscoco')
    # 加载数据集
    
    batch_loader = BatchLoader(datasets=datasets)
    # 生成batch_loader 对象
    parameters = Parameters(batch_loader.max_seq_len,
                            batch_loader.vocab_size)
    # 生成 参数 对象
    paraphraser = Paraphraser(parameters)
    # 生成复述器 对象
    ce_result_valid = []
    kld_result_valid = []
    ce_result_train = []
    kld_result_train = []
    ce_cur_train = []
    kld_cur_train = []

    if args.use_trained:
        paraphraser.load_state_dict(t.load('saved_models/trained_paraphraser_' + args.model_name))
        ce_result_valid = list(np.load('logs/ce_result_valid_{}.npy'.format(args.model_name)))
        kld_result_valid = list(np.load('logs/kld_result_valid_{}.npy'.format(args.model_name)))
        ce_result_train = list(np.load('logs/ce_result_train_{}.npy'.format(args.model_name)))
        kld_result_train = list(np.load('logs/kld_result_train_{}.npy'.format(args.model_name)))
        # load取出信息 
        # format格式化，这里应该是填充字符串 ：model_name
        
    if args.use_cuda:
        paraphraser = paraphraser.cuda()

    optimizer = Adam(paraphraser.learnable_parameters(), args.learning_rate, 
        weight_decay=args.weight_decay)

    train_step = paraphraser.trainer(optimizer, batch_loader)
    # 训练
    validate = paraphraser.validater(batch_loader)
    # 验证

    for iteration in range(args.num_iterations):    # 迭代 num_interations 次
       
        cross_entropy, kld, coef = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)

        ce_cur_train += [cross_entropy.data.cpu().numpy()]
        kld_cur_train += [kld.data.cpu().numpy()]

        # validation
        if iteration % 500 == 0:
            ce_result_train += [np.mean(ce_cur_train)]
            kld_result_train += [np.mean(kld_cur_train)]
            ce_cur_train, kld_cur_train = [], []

            print('\n')
            print('------------TRAIN-------------')
            print('----------ITERATION-----------')
            print(iteration)
            print('--------CROSS-ENTROPY---------')
            print(ce_result_train[-1])
            print('-------------KLD--------------')
            print(kld_result_train[-1])
            print('-----------KLD-coef-----------')
            print(coef)
            print('------------------------------')


            # averaging across several batches
            cross_entropy, kld = [], []
            for i in range(20):
                ce, kl, _ = validate(args.batch_size, args.use_cuda)
                cross_entropy += [ce.data.cpu().numpy()]
                kld += [kl.data.cpu().numpy()]
            
            kld = np.mean(kld)
            cross_entropy = np.mean(cross_entropy)
            ce_result_valid += [cross_entropy]
            kld_result_valid += [kld]

            print('\n')
            print('------------VALID-------------')
            print('--------CROSS-ENTROPY---------')
            print(cross_entropy)
            print('-------------KLD--------------')
            print(kld)
            print('------------------------------')

            _, _, (sampled, s1, s2) = validate(2, args.use_cuda, need_samples=True)
            
            for i in range(len(sampled)):
                result = paraphraser.sample_with_pair(batch_loader, 20, args.use_cuda, s1[i], s2[i])
                print('source: ' + s1[i])
                print('target: ' + s2[i])
                print('valid: ' + sampled[i])
                print('sampled: ' + result)
                print('...........................')

        # save model
        if (iteration % 10000 == 0 and iteration != 0) or iteration == (args.num_iterations - 1):
            t.save(paraphraser.state_dict(), 'saved_models/trained_paraphraser_' + args.model_name)
            np.save('logs/ce_result_valid_{}.npy'.format(args.model_name), np.array(ce_result_valid))
            np.save('logs/kld_result_valid_{}'.format(args.model_name), np.array(kld_result_valid))
            np.save('logs/ce_result_train_{}.npy'.format(args.model_name), np.array(ce_result_train))
            np.save('logs/kld_result_train_{}'.format(args.model_name), np.array(kld_result_train))

        # interm sampling 中间采样
        if (iteration % 20000 == 0 and iteration != 0) or iteration == (args.num_iterations - 1):
            if args.interm_sampling:
                SAMPLE_FILES = ['snli_test', 'mscoco_test', 'quora_test', 'snips']
                args.use_mean = False
                args.seq_len = 30
                
                for sample_file in SAMPLE_FILES:
                    result, target, source = sample.sample_with_input_file(batch_loader,
                                                paraphraser, args, sample_file)


                    sampled_file_dst = 'logs/intermediate/sampled_out_{}k_{}{}.txt'.format(
                                                iteration//1000, sample_file, args.model_name)
                    target_file_dst = 'logs/intermediate/target_out_{}k_{}{}.txt'.format(
                                                iteration//1000, sample_file, args.model_name)    
                    source_file_dst = 'logs/intermediate/source_out_{}k_{}{}.txt'.format(
                                                iteration//1000, sample_file, args.model_name)    
                    np.save(sampled_file_dst, np.array(result))
                    np.save(target_file_dst, np.array(target))
                    np.save(source_file_dst, np.array(source))
                    print('------------------------------')
                    print('results saved to: ')
                    print(sampled_file_dst)
                    print(target_file_dst)
                    print(source_file_dst)
            
