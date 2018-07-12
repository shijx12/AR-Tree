import argparse
import logging
import os
import time
from collections import defaultdict
from tensorboardX import SummaryWriter
from IPython import embed

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import softmax

from model.SingleModel import SingleModel
from model.PairModel import PairModel
from age.dataLoader import AGE2
from sst.dataLoader import SST
from snli.dataLoader import SNLI
from evaluate import eval_iter


def train_iter(args, batch, model, params, criterion, optimizer):
    model.train(True)
    model_arg, label = batch
    logits, supplements = model(**model_arg)
    label_pred = logits.max(1)[1]
    accuracy = torch.eq(label, label_pred).float().mean()
    num_correct = torch.eq(label, label_pred).long().sum()
    loss = criterion(input=logits, target=label)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(parameters=params, max_norm=5)
    optimizer.step()
    if args.debug:
        info = ''
        info += '\n ############################################################################ \n'
        # for i in range(len(supplements['tree'])):
        for i in range(5):
            info += '%s\n' % supplements['tree'][i]
        logging.info(info)
    return loss, accuracy


def train_rl_iter(args, batch, model, params, criterion, optimizer):
    model.train(True)
    model_arg, label = batch
    sample_num = args.sample_num
    logits, supplements = model(**model_arg)
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
    clip_grad_norm_(parameters=params, max_norm=5)
    optimizer.step()

    if args.debug:
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
            info += '%s\t%.2f, %.2f, %.2f\t%d\n' % (supplements['tree'][i], p[i][0].item(), p[i][1].item(), p[i][2].item(), label[i].item())
            for j in range(i*sample_num, (i+1)*sample_num):
                info += '%s\t%.2f, %.2f, %.2f\t%.2f\t%d\n' % (sample_trees[j], sample_p[j][0].item(), sample_p[j][1].item(), sample_p[j][2].item(), rl_rewards[j].item(), sample_label_gt[j].item())
            info += '%s\t%.2f, %.2f, %.2f\t%d\n' % (new_supplements['tree'][i], new_p[i][0].item(), new_p[i][1].item(), new_p[i][2].item(), label[i].item())
            info += ', '.join(map(lambda x: f'{x:.2f}', list(new_scores[i].squeeze().data)))+'\n'
            info += ' -------------------------------------------- \n'
        info += ' >>>\n >>>\n'
        wrong_mask = torch.ne(label, label_pred).data.cpu().numpy()
        wrong_i = [i for i in range(wrong_mask.shape[0]) if wrong_mask[i]==1]
        for i in wrong_i[:3]:
            info += ', '.join(map(lambda x: f'{x:.2f}', list(scores[i].squeeze().data)))+'\n'
            info += '%s\t%.2f, %.2f, %.2f\t%d\n' % (supplements['tree'][i], p[i][0].item(), p[i][1].item(), p[i][2].item(), label[i].item())
            for j in range(i*sample_num, (i+1)*sample_num):
                info += '%s\t%.2f, %.2f, %.2f\t%.2f\t%d\n' % (sample_trees[j], sample_p[j][0].item(), sample_p[j][1].item(), sample_p[j][2].item(), rl_rewards[j].item(), sample_label_gt[j].item())
            info += '%s\t%.2f, %.2f, %.2f\t%d\n' % (new_supplements['tree'][i], new_p[i][0].item(), new_p[i][1].item(), new_p[i][2].item(), label[i].item())
            info += ', '.join(map(lambda x: f'{x:.2f}', list(new_scores[i].squeeze().data)))+'\n'
            info += ' -------------------------------------------- \n'
        logging.info(info)
        
    return sv_loss, rl_loss, accuracy




def train(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    args.device = device

    ################################  data  ###################################
    if args.data_type == 'sst2':
        args.fine_grained = False
        data = SST(args) # some extra info will be appended into args
    elif args.data_type == 'sst5':
        args.fine_grained = True
        data = SST(args)
    elif args.data_type == 'age':
        data = AGE2(args)
    elif args.data_type == 'snli':
        data = SNLI(args)

    num_train_batches = data.num_train_batches # number of batches per epoch
    ################################  model  ###################################
    if args.data_type == 'snli':
        Model = PairModel
    else:
        Model = SingleModel
    model = Model(**vars(args))
    if args.glove:
        logging.info('Loading GloVe pretrained vectors...')
        model.word_embedding.weight.data.set_(data.weight)
    if args.fix_word_embedding:
        logging.info('Will not update word embeddings')
        model.word_embedding.weight.requires_grad = False
    model = model.to(device)
    logging.info(model)
    params = [p for p in model.parameters() if p.requires_grad]
    ################################################################

    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optimizer_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optimizer_class = optim.Adadelta
    optimizer = optimizer_class(params=params, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    criterion = nn.CrossEntropyLoss()
    trpack = [model, params, criterion, optimizer]

    train_summary_writer = SummaryWriter(os.path.join(args.save_dir, 'log', 'train'))
    valid_summary_writer = SummaryWriter(os.path.join(args.save_dir, 'log', 'valid'))

    logging.info(f'num_train_batches: {num_train_batches}')
    validate_every = num_train_batches // 10
    best_vaild_accuacy = 0
    iter_count = 0
    tic = time.time()

    for epoch_num in range(args.max_epoch):
        for batch_iter, train_batch in enumerate(data.train_minibatch_generator()):
            progress = epoch_num + batch_iter / num_train_batches
            iter_count += 1
            ################################# train iteration ####################################
            if args.model_type == 'Choi':
                train_loss, train_accuracy = train_iter(args, train_batch, *trpack)
                train_summary_writer.add_scalar('loss', train_loss, iter_count)
            ################################
            elif args.model_type == 'RL-SA':
                # args.rl_weight = initial_rl_weight / (1 + 100 * train_loader.epoch / args.max_epoch)
                train_sv_loss, train_rl_loss, train_accuracy = train_rl_iter(args, train_batch, *trpack)
                train_summary_writer.add_scalar('sv_loss', train_sv_loss, iter_count)
                train_summary_writer.add_scalar('rl_loss', train_rl_loss, iter_count)
            ################################
            elif args.model_type == 'STG-SA':
                train_loss, train_accuracy = train_iter(args, train_batch, *trpack)
                train_summary_writer.add_scalar('loss', train_loss, iter_count)
            ################################
            elif args.model_type == 'SSA':
                # model.encoder.temperature = 1 - 0.9 / args.max_epoch * int(progress) # 1 decrease to 0.1
                # model.encoder.attentree.temperature = 1 - 0.9 / args.max_epoch * int(progress)
                # print(model.encoder.attentree.temperature)
                train_loss, train_accuracy = train_iter(args, train_batch, *trpack)
                train_summary_writer.add_scalar('loss', train_loss, iter_count)
            else:
                raise Exception('unknown model')
            ########################################################################################
            if (batch_iter + 1) % (num_train_batches // 100) == 0:
                tac = (time.time() - tic) / 60
                print(f'   {tac:.2f} minutes\tprogress: {progress:.2f}, loss: {train_loss.item():.4f}')
            if (batch_iter + 1) % validate_every == 0:
                correct_sum = 0
                for valid_batch in data.dev_minibatch_generator():
                    correct, supplements = eval_iter(valid_batch, model)
                    correct_sum += correct
                valid_accuracy = correct_sum / data.num_valid
                scheduler.step(valid_accuracy)
                valid_summary_writer.add_scalar('acc', valid_accuracy, iter_count)
                logging.info(f'Epoch {progress:.2f}: '
                             f'valid accuracy = {valid_accuracy:.4f}')
                if valid_accuracy > best_vaild_accuacy:
                    correct_sum = 0
                    trees = []
                    for test_batch in data.test_minibatch_generator():
                        correct, supplements = eval_iter(test_batch, model)
                        correct_sum += correct
                        if 'tree' in supplements:
                            trees += supplements['tree']
                    test_accuracy = correct_sum / data.num_test
                    best_vaild_accuacy = valid_accuracy
                    model_filename = (f'model-{progress:.2f}'
                            f'-{valid_accuracy:.3f}'
                            f'-{test_accuracy:.3f}.pkl')
                    model_path = os.path.join(args.save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    print(f'Saved the new best model to {model_path}')



def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--data-type', required=True, choices=['sst2', 'sst5', 'age'])
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-type', required=True, choices=['Choi', 'RL-SA', 'tfidf', 'STG-SA', 'SSA'])


    parser.add_argument('--glove', default='glove.840B.300d')
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--cell-type', default='TriPad', choices=['Nary', 'TriPad'])
    parser.add_argument('--sample-num', default=3, type=int, help='sample num')
    parser.add_argument('--leaf-rnn-type', default='bilstm', choices=['no', 'bilstm'])
    parser.add_argument('--rank-input', default='h', choices=['w', 'h'], help='whether feed word embedding or hidden state of bilstm into score function')
    parser.add_argument('--rl_weight', default=0.1, type=float)
    parser.add_argument('--word-dim', default=300, type=int)


    parser.add_argument('--hidden-dim', type=int, help='dimension of final sentence embedding')
    parser.add_argument('--clf-hidden-dim', type=int)
    parser.add_argument('--clf-num-layers', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-epoch', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--l2reg', type=float)
    parser.add_argument('--optimizer')
    parser.add_argument('--fix-word-embedding', action='store_true')
    parser.add_argument('--use-batchnorm', action='store_true')
    parser.add_argument('--debug', action='store_true')

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
