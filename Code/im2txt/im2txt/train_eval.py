"""
Train and Evaluate the model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from im2txt import configuration

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--train_input_file_pattern', type=str, default='')
    parser.add_argument('--pretrained_model_checkpoint_file', type=str)
    parser.add_argument('--train_dir', type=str, default='')
    parser.add_argument('--train_CNN', type=bool, default=False)
    parser.add_argument('--number_of_steps', type=int, default=300000)
    parser.add_argument('--log_every_n_steps', type=int, default=1)
    # 数据集名称 Flickr8k Flickr30k MSCOCO
    parser.add_argument('--dataset_name', type=str, default='Flickr8k')
    # CNN模型名称 InceptionV3 InceptionV4 DenseNet ResNet
    parser.add_argument('--CNN_name', type=str, default='InceptionV3')

    # eval
    parser.add_argument('--eval_input_file_pattern', type=str, default='')
    # parser.add_argument('--eval_checkpoint_dir', type=str, default='') 直接用train_dir即可
    parser.add_argument('--eval_dir', type=str, default='')
    parser.add_argument('--eval_interval_secs', type=int, default=600)
    parser.add_argument('--num_eval_examples', type=int, default=10132)
    parser.add_argument('--min_global_step', type=int, default=5000)

    # Batch size,为0表示以cofiguration.py中为准.
    parser.add_argument('--batch_size', type=int, default=0)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

train_cmd = 'python ./train.py ' \
            '--input_file_pattern={input_file_pattern} ' \
            '--inception_checkpoint_file={inception_checkpoint_file} ' \
            '--train_dir={train_dir} ' \
            '--train_inception={train_inception} ' \
            '--number_of_steps={number_of_steps} ' \
            '--log_every_n_steps={log_every_n_steps} ' \
            '--CNN_name={CNN_name} ' \
            '--dataset_name={dataset_name} ' \
            '--batch_size={batch_size}'
eval_cmd = 'python ./evaluate.py ' \
           '--input_file_pattern={input_file_pattern} ' \
           '--checkpoint_dir={checkpoint_dir} ' \
           '--eval_dir={eval_dir} ' \
           '--eval_interval_secs={eval_interval_secs}  ' \
           '--num_eval_examples={num_eval_examples}  ' \
           '--min_global_step={min_global_step} ' \
           '--CNN_name={CNN_name} ' \
           '--batch_size={batch_size}'

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    print('current working dir [{0}]'.format(os.getcwd()))
    w_d = os.path.dirname(os.path.abspath(__file__))
    print('change wording dir to [{0}]'.format(w_d))
    os.chdir(w_d)

    model_config = configuration.ModelConfig()
    training_config = configuration.TrainingConfig()
    training_config.update_data_params(FLAGS.dataset_name)

    step_per_epoch = training_config.num_examples_per_epoch // model_config.batch_size
    epoch_num = FLAGS.number_of_steps // step_per_epoch
    print("Number of examples per epoch is", training_config.num_examples_per_epoch)
    print("Number of step per epoch is", step_per_epoch)
    print("To run", FLAGS.number_of_steps,"steps,run epoch number is", epoch_num)

    if FLAGS.pretrained_model_checkpoint_file:
        ckpt = ' --inception_checkpoint_file=' + FLAGS.pretrained_model_checkpoint_file
    else:
        ckpt = ''
    for i in range(epoch_num):
        steps = int(step_per_epoch * (i + 1))
        print('epoch{[', i, ']} goal steps :', steps)
        # train 1 epoch
        print('################    train    ################')
        p = os.popen(train_cmd.format(**{'input_file_pattern': FLAGS.train_input_file_pattern,
                                         'inception_checkpoint_file': FLAGS.pretrained_model_checkpoint_file,
                                         'train_dir': FLAGS.train_dir,
                                         'train_inception': FLAGS.train_CNN,
                                         'number_of_steps': steps,
                                         'log_every_n_steps': FLAGS.log_every_n_steps,
                                         'CNN_name': FLAGS.CNN_name,
                                         'dataset_name': FLAGS.dataset_name,
                                         'batch_size': FLAGS.batch_size}) + ckpt)
        for l in p:
            print(l.strip())

        # eval
        if steps < FLAGS.min_global_step:
            print('Global step = ', steps,' < ', FLAGS.min_global_step,', ignore eval this epoch!')
            continue
        print('################    eval    ################')
        p = os.popen(eval_cmd.format(**{'input_file_pattern': FLAGS.eval_input_file_pattern,
                                        'checkpoint_dir': FLAGS.train_dir,
                                        'eval_dir': FLAGS.eval_dir,
                                        'eval_interval_secs': FLAGS. eval_interval_secs,
                                        'num_eval_examples': FLAGS.num_eval_examples,
                                        'min_global_step': FLAGS.min_global_step,
                                        'CNN_name': FLAGS.CNN_name,
                                        'batch_size': FLAGS.batch_size}))
        for l in p:
            print(l.strip())

