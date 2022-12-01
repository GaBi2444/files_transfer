import os
import json

from reason.options.test_options import TestOptions
from reason.datasets import get_dataloader
from reason.models.parser import Seq2seqParser
from reason.models import get_vocab
import reason.utils.utils as utils
import pdb


def decode(idxs, idx_to_token):
    tokens = []
    for i in idxs:
        token = idx_to_token[i]
        if token in ['<END>', '<NULL>']:
            return tokens
        if token not in ['<START>', '<END>']:
            tokens.append(token)
    return tokens

opt = TestOptions().parse()

save_dir = os.path.dirname(opt.save_result_path)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
test_loader = get_dataloader(opt, opt.split, run_test=True)


model = Seq2seqParser(opt)
vocab = get_vocab(opt)
parse_results = {}
for x, y, ans, img_idx, q_idx in test_loader:
    model.set_input(x, y)
    pred_prog = model.parse()
    for i in range(pred_prog.size(0)):
        tmp_img_idx = int(img_idx[i])
        tmp_q_idx = int(q_idx[i])
        if tmp_img_idx not in parse_results:
            parse_results[tmp_img_idx] = {}
        if tmp_q_idx not in parse_results[tmp_img_idx]:
            parse_results[tmp_img_idx][tmp_q_idx] = []
        parse_results[tmp_img_idx][tmp_q_idx].append(decode(pred_prog[i].numpy(), vocab['program_idx_to_token']))
    print(img_idx[-1])

#pdb.set_trace()
print('saving results to %s' % opt.save_result_path)
with open(opt.save_result_path, 'w') as fout:
    json.dump(parse_results, fout)
