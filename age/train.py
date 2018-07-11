import argparse
import logging
import os
import time

import tensorboard

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm
from torch.nn.functional import softmax

from sst.model import SSTModel
from age.dataset import AGE2
from utils.helper import * 
import conf
from collections import defaultdict
from IPython import embed


def train_iter(args, batch, model, params, criterion, optimizer):
    model.train(True)
    words, length, label = batch
    logits, supplements = model(words=words, length=length)
    label_pred = logits.max(1)[1]
    accuracy = torch.eq(label, label_pred).float().mean()
    num_correct = torch.eq(label, label_pred).long().sum()
    loss = criterion(input=logits, target=label)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(parameters=params, max_norm=5)
    optimizer.step()
    if conf.debug:
        info = ''
        info += '\n ############################################################################ \n'
        # for i in range(len(supplements['tree'])):
        for i in range(5):
            info += '%s\n' % supplements['tree'][i]
        logging.info(info)
    return loss, accuracy


def train_rl_iter(args, batch, model, params, criterion, optimizer):
    model.train(True)
    sample_num = args.sample_num
    words, length, label = batch
    logits, supplements = model(words=words, length=length)
    label_pred = logits.max(1)[1]
    accuracy = torch.eq(label, label_pred).float().mean()
    num_correct = torch.eq(label, label_pred).long().sum()
    sv_loss = criterion(input=logits, target=label)
    ###########################
    # rl training loss for sampled trees
    sample_logits, probs, sample_trees = supplements['sample_logits'], supplements['probs'], supplements['sample_trees']
    sample_label_pred = sample_logits.max(1)[1]
    sample_label_gt = label.unsqueeze(1).expand(-1, sample_num).contiguous().view(-1)
    
    # hard reward
    rl_rewards = torch.eq(sample_label_gt, sample_label_pred).float().detach() * 2 - 1
    # soft reward
    '''rl_rewards = torch.gather(
            softmax(sample_logits, dim=1), 
            dim=1, 
            index=sample_label_gt.unsqueeze(1)
        ).float().squeeze(1).detach() * 2 - 1'''

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
        for i in range(7):
            info += ', '.join(map(lambda x: f'{x:.2f}', list(scores[i].squeeze().data)))+'\n'
            info += '%s\t%.2f, %.2f, %.2f\t%d\n' % (supplements['trees'][i], unwrap_scalar_variable(p[i][0]), unwrap_scalar_variable(p[i][1]), unwrap_scalar_variable(p[i][2]), unwrap_scalar_variable(label[i]))
            for j in range(i*sample_num, (i+1)*sample_num):
                info += '%s\t%.2f, %.2f, %.2f\t%.2f\t%d\n' % (sample_trees[j], unwrap_scalar_variable(sample_p[j][0]), unwrap_scalar_variable(sample_p[j][1]), unwrap_scalar_variable(sample_p[j][2]), unwrap_scalar_variable(rl_rewards[j]), unwrap_scalar_variable(sample_label_gt[j]))
            info += '%s\t%.2f, %.2f, %.2f\t%d\n' % (new_supplements['trees'][i], unwrap_scalar_variable(new_p[i][0]), unwrap_scalar_variable(new_p[i][1]), unwrap_scalar_variable(new_p[i][2]), unwrap_scalar_variable(label[i]))
            info += ', '.join(map(lambda x: f'{x:.2f}', list(new_scores[i].squeeze().data)))+'\n'
            info += ' -------------------------------------------- \n'
        info += ' >>>\n >>>\n'
        wrong_mask = torch.ne(label, label_pred).data.cpu().numpy()
        wrong_i = [i for i in range(wrong_mask.shape[0]) if wrong_mask[i]==1]
        for i in wrong_i[:3]:
            info += ', '.join(map(lambda x: f'{x:.2f}', list(scores[i].squeeze().data)))+'\n'
            info += '%s\t%.2f, %.2f, %.2f\t%d\n' % (supplements['trees'][i], unwrap_scalar_variable(p[i][0]), unwrap_scalar_variable(p[i][1]), unwrap_scalar_variable(p[i][2]), unwrap_scalar_variable(label[i]))
            for j in range(i*sample_num, (i+1)*sample_num):
                info += '%s\t%.2f, %.2f, %.2f\t%.2f\t%d\n' % (sample_trees[j], unwrap_scalar_variable(sample_p[j][0]), unwrap_scalar_variable(sample_p[j][1]), unwrap_scalar_variable(sample_p[j][2]), unwrap_scalar_variable(rl_rewards[j]), unwrap_scalar_variable(sample_label_gt[j]))
            info += '%s\t%.2f, %.2f, %.2f\t%d\n' % (new_supplements['trees'][i], unwrap_scalar_variable(new_p[i][0]), unwrap_scalar_variable(new_p[i][1]), unwrap_scalar_variable(new_p[i][2]), unwrap_scalar_variable(label[i]))
            info += ', '.join(map(lambda x: f'{x:.2f}', list(new_scores[i].squeeze().data)))+'\n'
            info += ' -------------------------------------------- \n'
        logging.info(info)
        
    return sv_loss, rl_loss, accuracy


def eval_iter(args, model, batch):
    model.train(False)
    words, length, label = batch
    logits, supplements = model(words=words, length=length)
    label_pred = logits.max(1)[1]
    num_correct = torch.eq(label, label_pred).long().sum()
    return num_correct, supplements 






def train(args):
    data = AGE2(datapath=args.data, batch_size=args.batch_size)
    num_classes = 5
    num_words = data.num_words
    ################################  model  ###################################
    if args.model_type == 'Choi':
        model = SSTModel(typ=args.model_type,
            num_classes=num_classes, num_words=num_words,
            word_dim=args.word_dim, hidden_dim=args.hidden_dim,
            clf_hidden_dim=args.clf_hidden_dim,
            clf_num_layers=args.clf_num_layers,
            use_leaf_rnn=args.leaf_rnn,
            bidirectional=args.bidirectional,
            use_batchnorm=args.batchnorm,
            dropout_prob=args.dropout,
            weighted_by_interval_length=args.weighted,
            weighted_base=args.weighted_base,
            weighted_update=args.weighted_update)
    elif args.model_type == 'RL-SA':
        model = SSTModel(typ=args.model_type,
            vocab=data,
            num_classes=num_classes, num_words=num_words,
            word_dim=args.word_dim, hidden_dim=args.hidden_dim,
            clf_hidden_dim=args.clf_hidden_dim,
            clf_num_layers=args.clf_num_layers,
            use_leaf_rnn=args.leaf_rnn,
            use_batchnorm=args.batchnorm,
            dropout_prob=args.dropout,
            bidirectional=args.bidirectional,
            cell_type=args.cell_type,
            sample_num=args.sample_num,
            rich_state=args.rich_state,
            rank_init=args.rank_init,
            rank_input=args.rank_input,
            rank_detach=(args.rank_detach==1))
    elif args.model_type == 'STG-SA':
        model = SNLIModel(typ=args.model_type,
            vocab=data,
            num_classes=num_classes, num_words=num_words,
            word_dim=args.word_dim, hidden_dim=args.hidden_dim,
            clf_hidden_dim=args.clf_hidden_dim,
            clf_num_layers=args.clf_num_layers,
            use_leaf_rnn=args.leaf_rnn,
            use_batchnorm=args.batchnorm,
            dropout_prob=args.dropout,
            bidirectional=args.bidirectional,
            cell_type=args.cell_type,
            rich_state=args.rich_state,
            rank_init=args.rank_init,
            rank_input=args.rank_input,
            rank_detach=(args.rank_detach==1))
    elif args.model_type == 'SSA':
        model = SSTModel(typ=args.model_type,
            vocab=data,
            num_classes=num_classes, num_words=num_words,
            word_dim=args.word_dim, hidden_dim=args.hidden_dim,
            clf_hidden_dim=args.clf_hidden_dim,
            clf_num_layers=args.clf_num_layers,
            use_batchnorm=args.batchnorm,
            dropout_prob=args.dropout,
            cell_type=args.cell_type,
            rich_state=args.rich_state,
            rank_init=args.rank_init,
            rank_detach=(args.rank_detach==1))
    ################################################################

    logging.info(model)
    if args.glove:
        logging.info('Loading GloVe pretrained vectors...')
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
    trpack = [model, params, criterion, optimizer]

    train_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'train'), flush_secs=10)
    valid_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'valid'), flush_secs=10)
    tsw, vsw = train_summary_writer, valid_summary_writer

    num_train_batches = data.train_size // data.batch_size 
    logging.info(f'num_train_batches: {num_train_batches}')
    validate_every = num_train_batches // 10
    best_vaild_accuacy = 0
    iter_count = 0
    tic = time.time()

    for epoch_num in range(args.max_epoch):
        for batch_iter, train_batch in enumerate(data.train_minibatch_generator()):
            progress = epoch_num + batch_iter/num_train_batches
            iter_count += 1
            ################################# train iteration ####################################
            if args.model_type == 'Choi':
                train_loss, train_accuracy = train_iter(args, train_batch, *trpack)
                add_scalar_summary(tsw, 'loss', train_loss, iter_count)
                add_scalar_summary(tsw, 'acc', train_accuracy, iter_count)
            ################################
            elif args.model_type == 'RL-SA':
                # args.rl_weight = initial_rl_weight / (1 + 100 * train_loader.epoch / args.max_epoch)
                train_sv_loss, train_rl_loss, train_accuracy = train_rl_iter(args, train_batch, *trpack)
                add_scalar_summary(tsw, 'sv_loss', train_sv_loss, iter_count)
                add_scalar_summary(tsw, 'rl_loss', train_rl_loss, iter_count)
                add_scalar_summary(tsw, 'acc', train_accuracy, iter_count)
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None and 'rank' in name:
                        add_histo_summary(tsw, 'value/'+name, p, iter_count)
                        add_histo_summary(tsw, 'grad/'+name, p.grad, iter_count)
            ################################
            elif args.model_type == 'STG-SA':
                train_loss, train_accuracy = train_iter(args, train_batch, *trpack)
                add_scalar_summary(tsw, 'loss', train_loss, iter_count)
                add_scalar_summary(tsw, 'acc', train_accuracy, iter_count)
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None and 'rank' in name:
                        add_histo_summary(tsw, 'value/'+name, p, iter_count)
                        add_histo_summary(tsw, 'grad/'+name, p.grad, iter_count)
            ################################
            elif args.model_type == 'SSA':
                # model.encoder.temperature = 1 - 0.9 / args.max_epoch * int(progress) # 1 decrease to 0.1
                print(model.encoder.temperature)
                train_loss, train_accuracy = train_iter(args, train_batch, *trpack)
                add_scalar_summary(tsw, 'loss', train_loss, iter_count)
                add_scalar_summary(tsw, 'acc', train_accuracy, iter_count)
                '''for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None and 'selfseglstm' in name:
                        add_histo_summary(tsw, 'value/'+name, p, iter_count)
                        add_histo_summary(tsw, 'grad/'+name, p.grad, iter_count)'''
            else:
                raise Exception('unknown model')
            ########################################################################################
            if (batch_iter + 1) % (num_train_batches // 100) == 0:
                tac = (time.time() - tic) / 60
                print(f'   {tac:.2f} minutes\tprogress: {progress:.2f}')
            if (batch_iter + 1) % validate_every == 0:
                correct_sum = 0
                for valid_batch in data.dev_minibatch_generator():
                    correct, supplements = eval_iter(args, model, valid_batch)
                    correct_sum += unwrap_scalar_variable(correct)
                valid_accuracy = correct_sum / data.dev_size 
                scheduler.step(valid_accuracy)
                add_scalar_summary(vsw, 'acc', valid_accuracy, iter_count)
                logging.info(f'Epoch {progress:.2f}: '
                             f'valid accuracy = {valid_accuracy:.4f}')
                if valid_accuracy > best_vaild_accuacy:
                    correct_sum = 0
                    trees = []
                    for test_batch in data.test_minibatch_generator():
                        correct, supplements = eval_iter(args, model, test_batch)
                        correct_sum += unwrap_scalar_variable(correct)
                        if 'tree' in supplements:
                            trees += supplements['tree']
                    test_accuracy = correct_sum / data.test_size
                    best_vaild_accuacy = valid_accuracy
                    model_filename = (f'model-{progress:.2f}'
                            f'-{valid_accuracy:.3f}'
                            f'-{test_accuracy:.3f}.pkl')
                    model_path = os.path.join(args.save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    print(f'Saved the new best model to {model_path}')
    # for t in trees:
    #     print(t)


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@') 
    parser.add_argument('--data', default='./data/age2.pickle')
    parser.add_argument('--glove', default='glove.840B.300d')
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--cell-type', default='TriPad', choices=['Nary', 'TriPad'])
    parser.add_argument('--model-type', default='Choi', choices=['Choi', 'RL-SA', 'tfidf', 'STG-SA', 'SSA'])
    parser.add_argument('--sample-num', default=3, type=int, help='sample num')
    parser.add_argument('--rich-state', default=False, action='store_true')
    parser.add_argument('--rank-init', default='kaiming', choices=['normal', 'kaiming'])
    parser.add_argument('--rank-input', default='word', choices=['word', 'h'])
    parser.add_argument('--rank-detach', default=0, choices=[0, 1], type=int, help='1 means detach, 0 means no detach')


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

    train(args)


if __name__ == '__main__':
    main()
