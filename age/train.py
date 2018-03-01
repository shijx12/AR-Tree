import argparse
import logging
import os

import tensorboard

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm
from torch.nn.functional import softmax

from sst.model import SSTModel, SSTAttModel
from age.dataset import AGE2
from utils.helper import * 
import conf
from collections import defaultdict
from IPython import embed


def train(args):
    data = AGE2(datapath=args.data, batch_size=args.batch_size)
    num_classes = 5
    num_words = data.num_words
    if args.model_type == 'binary':
        model = SSTModel(num_classes=num_classes, num_words=num_words,
                    word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                    clf_hidden_dim=args.clf_hidden_dim,
                    clf_num_layers=args.clf_num_layers,
                    use_leaf_rnn=args.leaf_rnn,
                    bidirectional=args.bidirectional,
                    use_batchnorm=args.batchnorm,
                    dropout_prob=args.dropout,
                    weighted_by_interval_length=args.weighted,
                    weighted_base=args.weighted_base,
                    weighted_update=args.weighted_update,
                    cell_type=args.cell_type)
    else:
        model = SSTAttModel(vocab=data,
                      num_classes=num_classes, num_words=num_words,
                      word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                      clf_hidden_dim=args.clf_hidden_dim,
                      clf_num_layers=args.clf_num_layers,
                      use_leaf_rnn=args.leaf_rnn,
                      use_batchnorm=args.batchnorm,
                      dropout_prob=args.dropout,
                      bidirectional=args.bidirectional,
                      cell_type=args.cell_type,
                      att_type=args.att_type,
                      sample_num=args.sample_num,
                      rich_state=args.rich_state)
    logging.info(model)
    if args.glove:
        model.word_embedding.weight.data.set_(data.weight)
    if args.fix_word_embedding:
        logging.info('Will not update word embeddings')
        model.word_embedding.weight.requires_grad = False
    if args.gpu > -1:
        logging.info(f'Using GPU {args.gpu}')
        model.cuda(args.gpu)
    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optimizer_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optimizer_class = optim.Adadelta
    elif args.optimizer == 'SGD':
        optimizer_class = optim.SGD
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_class(params=params, lr=args.lr, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    criterion = nn.CrossEntropyLoss()

    train_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'train'), flush_secs=10)
    valid_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'valid'), flush_secs=10)

    def run_iter(batch, is_training):
        model.train(is_training)
        words, length, label = batch
        length = wrap_with_variable(length, volatile=not is_training, gpu=args.gpu)
        words = wrap_with_variable(words, volatile=not is_training, gpu=args.gpu)
        label = wrap_with_variable(label, volatile=not is_training, gpu=args.gpu)
        logits, supplements = model(words=words, length=length, display=not is_training)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
        num_correct = torch.eq(label, label_pred).long().sum()
        loss = criterion(input=logits, target=label)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(parameters=params, max_norm=5)
            optimizer.step()
            return loss, accuracy
        else:
            if args.model_type == 'binary':
                select_masks = supplements['select_masks']
                depth_cumsum, word_cumsum = parse_tree_avg_depth(length, select_masks)
                return num_correct, depth_cumsum, word_cumsum
            elif args.model_type == 'att' and args.att_type == 'corpus':
                trees = supplements['trees']
                wrong_mask = torch.ne(label, label_pred).data.cpu().numpy()
                wrong_trees = [trees[i] for i in range(wrong_mask.shape[0]) if wrong_mask[i]==1]
                return num_correct, trees, wrong_trees

    num_train_batches = data.train_size // data.batch_size 
    logging.info(f'num_train_batches: {num_train_batches}')
    validate_every = num_train_batches // 10
    best_vaild_accuacy = 0
    iter_count = 0
    for epoch in range(args.max_epoch):
        for batch_iter, train_batch in enumerate(data.train_minibatch_generator()):
            train_loss, train_accuracy = run_iter(batch=train_batch, is_training=True)
            iter_count += 1
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='loss', value=train_loss, step=iter_count)
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='accuracy', value=train_accuracy, step=iter_count)

            if (batch_iter + 1) % validate_every == 0:
                valid_accuracy_sum = 0
                ###############################
                if args.model_type == 'binary':
                    depth_cumsum = word_cumsum = 0
                    for valid_batch in data.dev_minibatch_generator():
                        valid_correct, depth_, word_ = run_iter(batch=valid_batch, is_training=False)
                        valid_accuracy_sum += unwrap_scalar_variable(valid_correct)
                        depth_cumsum += depth_
                        word_cumsum += word_
                    valid_depth = depth_cumsum / word_cumsum
                ###############################
                elif args.model_type == 'att' and args.att_type == 'corpus':
                    wrong_trees = []
                    for valid_batch in data.dev_minibatch_generator():
                        valid_correct, _, wrong_ = run_iter(batch=valid_batch, is_training=False)
                        valid_accuracy_sum += unwrap_scalar_variable(valid_correct)
                        wrong_trees += wrong_
                    valid_depth = -1
                    # print some sample wrong trees
                    logging.info(' ***** sample wrong trees ***** ')
                    display = '\n'
                    for t in wrong_trees[:5]:
                        display += '%s\n' % (t)
                    logging.info(display + ' ***********************  ')
                ###############################
                valid_accuracy = valid_accuracy_sum / data.dev_size 
                add_scalar_summary(
                    summary_writer=valid_summary_writer,
                    name='accuracy', value=valid_accuracy, step=iter_count)
                scheduler.step(valid_accuracy)
                progress = iter_count / num_train_batches 
                logging.info(f'Epoch {progress:.2f}: '
                             f'valid accuracy = {valid_accuracy:.4f}')
                if valid_accuracy > best_vaild_accuacy:
                    #############################
                    # test performance
                    test_accuracy_sum = 0
                    for test_batch in data.test_minibatch_generator():
                        test_correct, _, _ = run_iter(batch=test_batch, is_training=False)
                        test_accuracy_sum += unwrap_scalar_variable(test_correct)
                    test_accuracy = test_accuracy_sum / data.test_size
                    ############################
                    best_vaild_accuacy = valid_accuracy
                    model_filename = (f'model-{progress:.2f}'
                            f'-{valid_accuracy:.4f}'
                            f'-{test_accuracy:.4f}.pkl')
                    model_path = os.path.join(args.save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    print(f'Saved the new best model to {model_path}')





def train_withsampleRL(args):
    data = AGE2(datapath=args.data, batch_size=args.batch_size)
    num_classes = 5 
    num_words = data.num_words 
    model = SSTAttModel(vocab=data,
                      num_classes=num_classes, num_words=num_words,
                      word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                      clf_hidden_dim=args.clf_hidden_dim,
                      clf_num_layers=args.clf_num_layers,
                      use_leaf_rnn=args.leaf_rnn,
                      use_batchnorm=args.batchnorm,
                      dropout_prob=args.dropout,
                      bidirectional=args.bidirectional,
                      cell_type=args.cell_type,
                      att_type=args.att_type,
                      sample_num=args.sample_num,
                      rich_state=args.rich_state,
                      rank_init=args.rank_init,
                      rank_input=args.rank_input,
                      rank_detach=(args.rank_detach==1),
                      rank_tanh=args.rank_tanh)
    logging.info(model)
    if args.glove:
        model.word_embedding.weight.data.set_(data.weight)
    if args.fix_word_embedding:
        logging.info('Will not update word embeddings')
        model.word_embedding.weight.requires_grad = False
    if args.gpu > -1:
        logging.info(f'Using GPU {args.gpu}')
        model.cuda(args.gpu)
    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optimizer_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optimizer_class = optim.Adadelta
    elif args.optimizer == 'SGD':
        optimizer_class = optim.SGD
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_class(params=params, lr=args.lr, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    criterion = nn.CrossEntropyLoss()

    train_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'train'), flush_secs=10)
    valid_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'valid'), flush_secs=10)

    def run_iter(batch, is_training):
        model.train(is_training)
        words, length, label = batch
        sample_num = args.sample_num

        length = wrap_with_variable(length, volatile=not is_training, gpu=args.gpu)
        words = wrap_with_variable(words, volatile=not is_training, gpu=args.gpu)
        label = wrap_with_variable(label, volatile=not is_training, gpu=args.gpu)
        logits, supplements = model(words=words, length=length, display=not is_training)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
        num_correct = torch.eq(label, label_pred).long().sum()
        sv_loss = criterion(input=logits, target=label)
        if is_training:
            ###########################
            # rl training loss for sampled trees
            sample_logits, probs, sample_trees = supplements['sample_logits'], supplements['probs'], supplements['sample_trees']
            sample_label_pred = sample_logits.max(1)[1]
            sample_label_gt = label.unsqueeze(1).expand(-1, sample_num).contiguous().view(-1)
            
            rl_rewards = torch.eq(sample_label_gt, sample_label_pred).float().detach() * 2 - 1
            rl_loss = 0
            
            # average of word
            final_probs = defaultdict(list)
            for i in range(len(words)):
                cand_rewards = rl_rewards[i*sample_num: (i+1)*sample_num]
                for j in range(sample_num):
                    k = i * sample_num + j
                    for w in probs[k]:
                        final_probs[w] += [p*rl_rewards[k] for p in probs[k][w]]
            for w in final_probs:
                rl_loss += - sum(final_probs[w]) / len(final_probs[w])
            if len(final_probs) > 0:
                rl_loss /= len(final_probs)
            rl_loss *= args.rl_weight
            ###########################
            total_loss = sv_loss + rl_loss
            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm(parameters=params, max_norm=5)
            optimizer.step()

            if conf.debug:
                info = ''
                info += '\n ############################################################################ \n'
                p = softmax(logits, dim=1)
                scores = model.encoder.scores
                sample_p = softmax(sample_logits, dim=1)
                logits, new_supplements = model(words=words, length=length, display=True)
                new_p = softmax(logits, dim=1)
                new_scores = model.encoder.scores
                for i in range(1):
                    info += ', '.join(map(lambda x: f'{x:.2f}', list(scores[i].squeeze().data)))+'\n'
                    info += '%s\t%.1f, %.1f, %.1f, %.1f, %.1f\t%d\n' % (supplements['trees'][i], unwrap_scalar_variable(p[i][0]), unwrap_scalar_variable(p[i][1]), unwrap_scalar_variable(p[i][2]), unwrap_scalar_variable(p[i][3]), unwrap_scalar_variable(p[i][4]), unwrap_scalar_variable(label[i]))
                    for j in range(i*sample_num, (i+1)*sample_num):
                        info += '%s\t%.1f, %.1f, %.1f, %.1f, %.1f\t%.2f\t%d\n' % (sample_trees[j], unwrap_scalar_variable(sample_p[j][0]), unwrap_scalar_variable(sample_p[j][1]), unwrap_scalar_variable(sample_p[j][2]), unwrap_scalar_variable(sample_p[j][3]), unwrap_scalar_variable(sample_p[j][4]), unwrap_scalar_variable(rl_rewards[j]), unwrap_scalar_variable(sample_label_gt[j]))
                    info += '%s\t%.1f, %.1f, %.1f, %.1f, %.1f\t%d\n' % (new_supplements['trees'][i], unwrap_scalar_variable(new_p[i][0]), unwrap_scalar_variable(new_p[i][1]), unwrap_scalar_variable(new_p[i][2]), unwrap_scalar_variable(new_p[i][3]), unwrap_scalar_variable(new_p[i][4]), unwrap_scalar_variable(label[i]))
                    info += ', '.join(map(lambda x: f'{x:.2f}', list(new_scores[i].squeeze().data)))+'\n'
                    info += ' -------------------------------------------- \n'
                logging.info(info)
                

            return sv_loss, rl_loss, accuracy
        else:
            trees = supplements['trees']
            wrong_mask = torch.ne(label, label_pred).data.cpu().numpy()
            wrongs = [{'tree': trees[i],
                'label': unwrap_scalar_variable(label[i]),
                'label_pred': unwrap_scalar_variable(label_pred[i])} for i in range(wrong_mask.shape[0]) if wrong_mask[i]==1]
            return num_correct, trees, wrongs

    num_train_batches = data.train_size // data.batch_size 
    logging.info(f'num_train_batches: {num_train_batches}')
    validate_every = num_train_batches // 10
    best_vaild_accuacy = 0
    iter_count = 0

    for epoch in range(args.max_epoch):
        for batch_iter, train_batch in enumerate(data.train_minibatch_generator()):
            train_sv_loss, train_rl_loss, train_accuracy = run_iter(batch=train_batch, is_training=True)
            iter_count += 1
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='sv_loss', value=train_sv_loss, step=iter_count)
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='rl_loss', value=train_rl_loss, step=iter_count)
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='accuracy', value=train_accuracy, step=iter_count)
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None and 'rank' in name:
                    add_histo_summary(
                        summary_writer=train_summary_writer,
                        name='value/'+name, value=p, step=iter_count)
                    add_histo_summary(
                        summary_writer=train_summary_writer,
                        name='grad/'+name, value=p.grad, step=iter_count)

            if (batch_iter + 1) % validate_every == 0:
                valid_accuracy_sum = 0
                ###############################
                wrongs = []
                for valid_batch in data.dev_minibatch_generator():
                    valid_correct, _, wrong_ = run_iter(batch=valid_batch, is_training=False)
                    valid_accuracy_sum += unwrap_scalar_variable(valid_correct)
                    wrongs += wrong_
                ###############################
                valid_accuracy = valid_accuracy_sum / data.dev_size 
                add_scalar_summary(
                    summary_writer=valid_summary_writer,
                    name='accuracy', value=valid_accuracy, step=iter_count)
                scheduler.step(valid_accuracy)
                progress = iter_count / num_train_batches
                logging.info(f'Epoch {progress:.2f}: '
                             f'valid accuracy = {valid_accuracy:.4f}')
                if valid_accuracy > best_vaild_accuacy:
                    #############################
                    # test performance
                    test_accuracy_sum = 0
                    for test_batch in data.test_minibatch_generator():
                        test_correct, _, _ = run_iter(batch=test_batch, is_training=False)
                        test_accuracy_sum += unwrap_scalar_variable(test_correct)
                    test_accuracy = test_accuracy_sum / data.test_size
                    ############################
                    best_vaild_accuacy = valid_accuracy
                    model_filename = (f'model-{progress:.2f}'
                            f'-{valid_accuracy:.4f}'
                            f'-{test_accuracy:.4f}.pkl')
                    model_path = os.path.join(args.save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    print(f'Saved the new best model to {model_path}')
    logging.info(' ***** all wrong trees in validation set ***** ')
    display = '\n'
    for t in wrongs:
        display += '%s\n' % (t)
    logging.info(display + ' ***********************  ')


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@') 
    parser.add_argument('--data', default='./data/age2.pickle')
    parser.add_argument('--glove', default='glove.840B.300d')
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--cell_type', default='treelstm', choices=['treelstm', 'Nary', 'TriPad'])
    parser.add_argument('--model_type', default='binary', choices=['binary', 'att'])
    parser.add_argument('--att_type', default='corpus', choices=['corpus', 'rank0', 'rank1', 'rank2'], help='Used only when model_type==att')
    parser.add_argument('--sample_num', default=3, type=int, help='sample num')
    parser.add_argument('--rich-state', default=False, action='store_true')
    parser.add_argument('--rank_init', default='kaiming', choices=['normal', 'kaiming'])
    parser.add_argument('--rank_input', default='word', choices=['word', 'h'])
    parser.add_argument('--rank_detach', default=1, choices=[0, 1], type=int, help='1 means detach, 0 means no detach')
    parser.add_argument('--rank_tanh', action='store_true')


    parser.add_argument('--rl_weight', default=0.1, type=float)
    parser.add_argument('--use_important_words', default=False, action='store_true')
    parser.add_argument('--important_words', default=['no','No','NO','not','Not','NOT','isn\'t','aren\'t','hasn\'t','haven\'t','can\'t'])
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--hidden-dim', default=300, type=int)
    parser.add_argument('--clf-hidden-dim', default=2000, type=int)
    parser.add_argument('--clf-num-layers', default=2, type=int)
    parser.add_argument('--leaf-rnn', default=True, action='store_true')
    parser.add_argument('--bidirectional', default=True, action='store_true')
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--fix-word-embedding', default=False, action='store_true')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--max-epoch', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--l2reg', default=1e-5, type=float)

    parser.add_argument('--weighted', default=False, action='store_true')
    parser.add_argument('--weighted_base', type=float, default=2)
    parser.add_argument('--weighted_update', default=False, action='store_true')
    args = parser.parse_args()

    #######################################
    # a simple log file, the same content as stdout
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'stdout.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    ########################################

    for k, v in vars(args).items():
        logging.info(k+':'+str(v))

    if args.att_type == 'corpus':
        train(args)
    else:
        train_withsampleRL(args)


if __name__ == '__main__':
    main()
