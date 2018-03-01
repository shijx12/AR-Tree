import argparse
import logging
import os

import tensorboard

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm
from torch.nn.functional import softmax
from torchtext import data, datasets

from sst.model import SSTModel, SSTAttModel
from utils.glove import load_glove
from utils.helper import * 
from collections import defaultdict
import conf
from IPython import embed

import random
random.seed(666)

def train(args):
    text_field = data.Field(lower=args.lower, include_lengths=True,
                            batch_first=True)
    label_field = data.Field(sequential=False)

    filter_pred = lambda ex: len(ex.text) < 500
    dataset_splits = datasets.IMDB.splits(
        root=args.datadir, text_field=text_field, label_field=label_field, filter_pred=filter_pred)

    text_field.build_vocab(*dataset_splits, vectors=args.glove)
    label_field.build_vocab(*dataset_splits)

    logging.info(f'Initialize with pretrained vectors: {args.glove}')
    logging.info(f'Number of classes: {len(label_field.vocab)}')

    if args.valid: # construct validation set
        train_set, test_set = dataset_splits
        # ratio of train / valid
        l = int(len(train_set) * 0.9)
        examples = train_set.examples
        random.shuffle(examples)
        train_set.examples = examples[:l]
        valid_set = datasets.IMDB(path=os.path.join(args.datadir, 'imdb', 'aclImdb', 'train'), text_field=text_field, label_field=label_field, filter_pred=filter_pred)
        valid_set.examples = examples[l:]
        dataset_splits = (train_set, valid_set, test_set)
        train_loader, valid_loader, _ = data.BucketIterator.splits(
            datasets=dataset_splits, batch_size=args.batch_size, device=args.gpu)
    else: # use test set as valid set
        train_loader, valid_loader = data.BucketIterator.splits(
            datasets=dataset_splits, batch_size=args.batch_size, device=args.gpu)


    num_classes = len(label_field.vocab)
    if args.model_type == 'binary':
        model = SSTModel(num_classes=num_classes, num_words=len(text_field.vocab),
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
        model = SSTAttModel(vocab=text_field.vocab,
                      num_classes=num_classes, num_words=len(text_field.vocab),
                      word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                      clf_hidden_dim=args.clf_hidden_dim,
                      clf_num_layers=args.clf_num_layers,
                      use_leaf_rnn=args.leaf_rnn,
                      use_batchnorm=args.batchnorm,
                      dropout_prob=args.dropout,
                      bidirectional=args.bidirectional,
                      cell_type=args.cell_type,
                      att_type=args.att_type,
                      sample_num=args.sample_num)
    logging.info(model)
    if args.glove:
        model.word_embedding.weight.data.set_(text_field.vocab.vectors)
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
    elif args.optimizer == 'sgd':
        optimizer_class = optim.SGD
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_class(params=params, lr=args.lr, weight_decay=args.l2reg)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.lrd_every_epoch)

    criterion = nn.CrossEntropyLoss()

    train_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'train'), flush_secs=10)
    valid_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'valid'), flush_secs=10)

    def run_iter(batch, is_training):
        model.train(is_training)
        words, length = batch.text
        label = batch.label
        length = wrap_with_variable(length, volatile=not is_training, gpu=args.gpu)
        logits, supplements = model(words=words, length=length, display=not is_training)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
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
                return loss, accuracy, depth_cumsum, word_cumsum
            elif args.model_type == 'att' and args.att_type == 'corpus':
                trees = supplements['trees']
                wrong_mask = torch.ne(label, label_pred).data.cpu().numpy()
                wrong_trees = [trees[i] for i in range(wrong_mask.shape[0]) if wrong_mask[i]==1]
                return loss, accuracy, trees, wrong_trees

    num_train_batches = len(train_loader)
    logging.info(f'num_train_batches: {num_train_batches}')
    validate_every = num_train_batches // 10
    iter_count = 0


    for batch_iter, train_batch in enumerate(train_loader):
        print(batch_iter)
        train_loss, train_accuracy = run_iter(batch=train_batch, is_training=True)
        iter_count += 1
        add_scalar_summary(
            summary_writer=train_summary_writer,
            name='loss', value=train_loss, step=iter_count)
        add_scalar_summary(
            summary_writer=train_summary_writer,
            name='accuracy', value=train_accuracy, step=iter_count)
        '''for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and 'embedding' not in name:
                add_histo_summary(
                    summary_writer=train_summary_writer,
                    name='value/'+name, value=p, step=iter_count)
                add_histo_summary(
                    summary_writer=train_summary_writer,
                    name='grad/'+name, value=p.grad, step=iter_count)'''

        if (batch_iter + 1) % validate_every == 0:
            valid_loss_sum = valid_accuracy_sum = 0
            num_valid_batches = len(valid_loader)
            ###############################
            if args.model_type == 'binary':
                depth_cumsum = word_cumsum = 0
                for valid_batch in valid_loader:
                    valid_loss, valid_accuracy, depth_, word_ = run_iter(batch=valid_batch, is_training=False)
                    valid_loss_sum += unwrap_scalar_variable(valid_loss)
                    valid_accuracy_sum += unwrap_scalar_variable(valid_accuracy)
                    depth_cumsum += depth_
                    word_cumsum += word_
                valid_depth = depth_cumsum / word_cumsum
            ###############################
            elif args.model_type == 'att' and args.att_type == 'corpus':
                wrong_trees = []
                for valid_batch in valid_loader:
                    valid_loss, valid_accuracy, _, wrong_ = run_iter(batch=valid_batch, is_training=False)
                    valid_loss_sum += unwrap_scalar_variable(valid_loss)
                    valid_accuracy_sum += unwrap_scalar_variable(valid_accuracy)
                    wrong_trees += wrong_
                valid_depth = -1
                # print some sample wrong trees
                logging.info(' ***** sample wrong trees ***** ')
                display = '\n'
                for t in wrong_trees[:5]:
                    display += '%s\n' % (t)
                logging.info(display + ' ***********************  ')
            ###############################
            valid_loss = valid_loss_sum / num_valid_batches
            valid_accuracy = valid_accuracy_sum / num_valid_batches
            add_scalar_summary(
                summary_writer=valid_summary_writer,
                name='loss', value=valid_loss, step=iter_count)
            add_scalar_summary(
                summary_writer=valid_summary_writer,
                name='accuracy', value=valid_accuracy, step=iter_count)
            scheduler.step()
            progress = train_loader.epoch
            logging.info(f'Epoch {progress:.2f}: '
                         f'valid accuracy = {valid_accuracy:.4f}, ')
            if progress > args.max_epoch:
                break
    model_filename = (f'-{valid_accuracy:.4f}.pkl')
    model_path = os.path.join(args.save_dir, model_filename)
    torch.save(model.state_dict(), model_path)


def train_withsampleRL(args):
    text_field = data.Field(lower=args.lower, include_lengths=True,
                            batch_first=True)
    label_field = data.Field(sequential=False)

    filter_pred = lambda ex: len(ex.text) < 500
    dataset_splits = datasets.IMDB.splits(
        root=args.datadir, text_field=text_field, label_field=label_field)

    text_field.build_vocab(*dataset_splits, vectors=args.glove)
    label_field.build_vocab(*dataset_splits)

    logging.info(f'Initialize with pretrained vectors: {args.glove}')
    logging.info(f'Number of classes: {len(label_field.vocab)}')

    if args.valid: # construct validation set
        train_set, test_set = dataset_splits
        # ratio of train / valid
        l = int(len(train_set) * 0.9)
        examples = train_set.examples
        random.shuffle(examples)
        train_set.examples = examples[:l]
        valid_set = datasets.IMDB(path=os.path.join(args.datadir, 'imdb', 'aclImdb', 'train'), text_field=text_field, label_field=label_field, filter_pred=filter_pred)
        valid_set.examples = examples[l:]
        dataset_splits = (train_set, valid_set, test_set)
        train_loader, valid_loader, _ = data.BucketIterator.splits(
            datasets=dataset_splits, batch_size=args.batch_size, device=args.gpu)
    else: # use test set as valid set
        train_loader, valid_loader = data.BucketIterator.splits(
            datasets=dataset_splits, batch_size=args.batch_size, device=args.gpu)

    num_classes = len(label_field.vocab)
    text_field.vocab.id_to_word = lambda i: text_field.vocab.itos[i]
    text_field.vocab.id_to_tf = lambda i: text_field.freqs[i]
    model = SSTAttModel(vocab=text_field.vocab,
                      num_classes=num_classes, num_words=len(text_field.vocab),
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
        model.word_embedding.weight.data.set_(text_field.vocab.vectors)
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
    elif args.optimizer == 'sgd':
        optimizer_class = optim.SGD
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_class(params=params, lr=args.lr, weight_decay=args.l2reg)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.lrd_every_epoch)

    criterion = nn.CrossEntropyLoss()

    train_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'train'), flush_secs=10)
    valid_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'valid'), flush_secs=10)

    def run_iter(batch, is_training):
        model.train(is_training)
        words, length = batch.text
        label = batch.label
        sample_num = args.sample_num

        length = wrap_with_variable(length, volatile=not is_training, gpu=args.gpu)
        logits, supplements = model(words=words, length=length, display=not is_training)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
        sv_loss = criterion(input=logits, target=label)
        if is_training:
            ###########################
            # rl training loss for sampled trees
            sample_logits, probs, sample_trees = supplements['sample_logits'], supplements['probs'], supplements['sample_trees']
            sample_label_pred = sample_logits.max(1)[1]
            sample_label_gt = label.unsqueeze(1).expand(-1, sample_num).contiguous().view(-1)
            # soft reward
            rl_rewards = torch.gather(
                    softmax(sample_logits, dim=1), 
                    dim=1, 
                    index=sample_label_gt.unsqueeze(1)
                ).float().squeeze(1).detach() * 2 - 1
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

            conf.debug = False
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
        else:
            trees = supplements['trees']
            wrong_mask = torch.ne(label, label_pred).data.cpu().numpy()
            wrongs = [{'tree': trees[i],
                'label': unwrap_scalar_variable(label[i]),
                'label_pred': unwrap_scalar_variable(label_pred[i])} for i in range(wrong_mask.shape[0]) if wrong_mask[i]==1]
            return sv_loss, accuracy, trees, wrongs


    num_train_batches = len(train_loader)
    logging.info(f'num_train_batches: {num_train_batches}')
    validate_every = num_train_batches // 10
    best_vaild_accuacy = 0
    iter_count = 0
    model_path = None
    for batch_iter, train_batch in enumerate(train_loader):
        print(batch_iter)
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

        if (batch_iter + 1) % validate_every == 0:
            valid_loss_sum = valid_accuracy_sum = 0
            num_valid_batches = len(valid_loader)
            ###############################
            wrong_trees = []
            for valid_batch in valid_loader:
                valid_loss, valid_accuracy, _, wrong_ = run_iter(batch=valid_batch, is_training=False)
                valid_loss_sum += unwrap_scalar_variable(valid_loss)
                valid_accuracy_sum += unwrap_scalar_variable(valid_accuracy)
                wrong_trees += wrong_
            # print some sample wrong trees
            ###############################
            valid_loss = valid_loss_sum / num_valid_batches
            valid_accuracy = valid_accuracy_sum / num_valid_batches
            add_scalar_summary(
                summary_writer=valid_summary_writer,
                name='accuracy', value=valid_accuracy, step=iter_count)
            scheduler.step()
            progress = train_loader.epoch
            logging.info(f'Epoch {progress:.2f}: '
                         f'valid accuracy = {valid_accuracy:.4f}, ')
            if valid_accuracy > best_vaild_accuacy:
                if model_path: # only preserve the best one
                    os.remove(model_path)
                best_vaild_accuacy = valid_accuracy
                model_filename = (f'model-{progress:.2f}'
                                  f'-{valid_accuracy:.4f}.pkl')
                model_path = os.path.join(args.save_dir, model_filename)
                torch.save(model.state_dict(), model_path)
                print(f'Saved the new best model to {model_path}')
            if progress > args.max_epoch:
                break
    logging.info(' ***** all wrong trees in validation set ***** ')
    display = '\n'
    for t in wrong_trees:
        display += '%s\n' % (t)
    logging.info(display + ' ***********************  ')


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@') 
    parser.add_argument('--datadir', default='/data/share/')
    parser.add_argument('--glove', default='glove.840B.300d')
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--valid', default=False, action='store_true')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--cell_type', default='treelstm', choices=['treelstm', 'simple', 'P2K', 'Tri', 'TriPad'])
    parser.add_argument('--model_type', default='binary', choices=['binary', 'att'])
    parser.add_argument('--att_type', default='corpus', choices=['corpus', 'rank0', 'rank1', 'rank2'], help='Used only when model_type==att')
    parser.add_argument('--sample_num', default=1, type=int)
    parser.add_argument('--rl_weight', default=0.1, type=float)
    parser.add_argument('--use_important_words', default=False, action='store_true')
    parser.add_argument('--important_words', default=['no','No','NO','not','Not','NOT','isn\'t','aren\'t','hasn\'t','haven\'t','can\'t'])

    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--hidden-dim', default=300, type=int)
    parser.add_argument('--clf-hidden-dim', default=1024, type=int)
    parser.add_argument('--clf-num-layers', default=2, type=int)
    parser.add_argument('--leaf-rnn', default=True, action='store_true')
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--fix-word-embedding', default=False, action='store_true')
    parser.add_argument('--rich-state', default=False, action='store_true')
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--max-epoch', default=20, type=int)
    parser.add_argument('--lr', default=1, type=float)
    parser.add_argument('--lrd-every-epoch', default=1, type=float)
    parser.add_argument('--optimizer', default='adadelta')
    parser.add_argument('--l2reg', default=1e-5, type=float)

    parser.add_argument('--weighted', default=False, action='store_true')
    parser.add_argument('--weighted_base', type=float, default=2)
    parser.add_argument('--weighted_update', default=False, action='store_true')

    parser.add_argument('--fine-grained', default=False, action='store_true')
    parser.add_argument('--lower', default=False, action='store_true')
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
