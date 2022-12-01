import json
import torch
import reason.utils.utils as utils
from reason.models import get_vocab
import pdb

class Trainer():
    """Trainer"""

    def __init__(self, opt, train_loader, val_loader, model, executor=None):
        self.opt = opt
        self.reinforce = opt.reinforce
        self.reward_decay = opt.reward_decay
        self.entropy_factor = opt.entropy_factor
        self.num_iters = opt.num_iters
        self.run_dir = opt.run_dir
        self.display_every = opt.display_every
        self.checkpoint_every = opt.checkpoint_every
        self.visualize_training = opt.visualize_training
        self.vocab = get_vocab(opt)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.executor = executor
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.seq2seq.parameters()),
                                          lr=opt.learning_rate)

        self.stats = {
            'train_losses': [],
            'train_batch_accs': [],
            'train_accs_ts': [],
            'val_losses': [],
            'val_accs': [],
            'val_accs_ts': [],
            'best_val_acc': -1,
            'model_t': 0
        }
        if opt.visualize_training:
            from reason.utils.logger import Logger
            self.logger = Logger('%s/logs' % opt.run_dir)

    def train(self):
        training_mode = 'reinforce' if self.reinforce else 'seq2seq'
        print('| start training %s, running in directory %s' % (training_mode, self.run_dir))
        t = 0
        epoch = 0
        baseline = 0
        #pdb.set_trace()
        while t < self.num_iters:
            epoch += 1
            for x, y, ans, idx in self.train_loader:
                t += 1
                loss, reward = None, None
                self.model.set_input(x, y)
                self.optimizer.zero_grad()
                if self.reinforce:
                    pred = self.model.reinforce_forward()
                    reward = self.get_batch_reward(pred, ans, idx, 'train')
                    baseline = reward * (1 - self.reward_decay) + baseline * self.reward_decay
                    advantage = reward - baseline
                    self.model.set_reward(advantage)
                    self.model.reinforce_backward(self.entropy_factor)
                else:
                    loss = self.model.supervised_forward()
                    self.model.supervised_backward()
                self.optimizer.step()

                if t % self.display_every == 0:
                    if self.reinforce:
                        self.stats['train_batch_accs'].append(reward)
                        self.log_stats('training batch reward', reward, t)
                        print('| iteration %d / %d, epoch %d, reward %f' % (t, self.num_iters, epoch, reward))
                    else:
                        self.stats['train_losses'].append(loss)
                        self.log_stats('training batch loss', loss, t)
                        print('| iteration %d / %d, epoch %d, loss %f' % (t, self.num_iters, epoch, loss))
                    self.stats['train_accs_ts'].append(t)

                if t % self.checkpoint_every == 0 or t >= self.num_iters:
                    print('| checking validation accuracy')
                    val_acc = self.check_val_accuracy()
                    print('| validation accuracy %f' % val_acc)
                    if val_acc >= self.stats['best_val_acc']:
                        print('| best model')
                        self.stats['best_val_acc'] = val_acc
                        self.stats['model_t'] = t
                        self.model.save_checkpoint('%s/checkpoint_best.pt' % self.run_dir)
                        self.model.save_checkpoint('%s/checkpoint_iter%08d.pt' % (self.run_dir, t))
                    if not self.reinforce:
                        val_loss = self.check_val_loss()
                        print('| validation loss %f' % val_loss)
                        self.stats['val_losses'].append(val_loss)
                        self.log_stats('val loss', val_loss, t)
                    self.stats['val_accs'].append(val_acc)
                    self.log_stats('val accuracy', val_acc, t)
                    self.stats['val_accs_ts'].append(t)
                    self.model.save_checkpoint('%s/checkpoint.pt' % self.run_dir)
                    with open('%s/stats.json' % self.run_dir, 'w') as fout:
                        save_dict = {}
                        for key_id, val_info in self.stats.items():
                            if isinstance(val_info, torch.Tensor):
                                val_info = float(val_info)
                            elif isinstance(val_info, list):
                                for e_idx, ele in enumerate(val_info):
                                    if isinstance(ele, torch.Tensor):
                                        val_info[e_idx] = float(ele)
                            self.stats[key_id] = val_info
                        json.dump(self.stats, fout)
                    self.log_params(t)

                if t >= self.num_iters:
                    break

    def check_val_loss(self):
        loss = 0
        t = 0
        for x, y, _, _ in self.val_loader:
            self.model.set_input(x, y)
            loss += self.model.supervised_forward()
            t += 1
        return loss / t if t is not 0 else 0

    # def check_val_accuracy(self):
    #     reward = 0
    #     t = 0
    #     for x, y, ans, idx in self.val_loader:
    #         self.model.set_input(x, y)
    #         pred = self.model.parse()
    #         reward += self.get_batch_reward(pred, ans, idx, 'val')
    #         t += 1
    #     reward = reward / t if t is not 0 else 0
    #     return reward

    def check_val_accuracy(self):
        acc = 0
        t = 0
        for x, y, ans, idx in self.val_loader:
            self.model.set_input(x, y)
            pred = self.model.parse()
            acc += self.match_program(pred, y)
            t += 1
        acc = acc / t if t is not 0 else 0
        return acc

    def match_program(self, pg_pred, pg_gt):
        correct, total = 0, 0
        for i in range(pg_pred.size(0)):
            pred_list, gt_list = [], []
            for m in pg_pred[i]:
                if m == self.vocab['program_token_to_idx']['<END>'] or \
                   m == self.vocab['program_token_to_idx']['<NULL>']:
                    break
                if m != self.vocab['program_token_to_idx']['<START>']:
                    pred_list.append(m)
            for m in pg_gt[i]:
                if m == self.vocab['program_token_to_idx']['<END>'] or \
                   m == self.vocab['program_token_to_idx']['<NULL>']:
                    break
                if m != self.vocab['program_token_to_idx']['<START>']:
                    gt_list.append(m)
            if pred_list == gt_list:
                correct += 1
            total += 1
        return float(correct) / total

    def get_batch_reward(self, programs, answers, image_idxs, split):
        reward = 0
        for i in range(programs.size(0)):
            pred = self.executor.run(programs[i], image_idxs[i], split)
            ans = self.vocab['answer_idx_to_token'][answers[i]]
            if pred == ans:
                reward += 1.0
        reward /= programs.size(0)
        return reward

    def log_stats(self, tag, value, t):
        if self.visualize_training and self.logger is not None:
            self.logger.scalar_summary(tag, value, t)

    def log_params(self, t):
        if self.visualize_training and self.logger is not None:
            for tag, value in self.model.seq2seq.named_parameters():
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag, self._to_numpy(value), t)
                if value.grad is not None:
                    self.logger.histo_summary('%s/grad' % tag, self._to_numpy(value.grad), t)

    def _to_numpy(self, x):
        return x.data.cpu().numpy()
