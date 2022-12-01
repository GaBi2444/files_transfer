import os
import json
import argparse

import h5py
import numpy as np
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--question_path', type=str, required=True)
parser.add_argument('--output_h5_path', type=str, required=True)
parser.add_argument('--question_type', type=str, choices=['descriptive'])#['mc_question', 'mc_choice', 'oe_question', 'retrieval', 'grounding'])
parser.add_argument('--input_vocab_path', type=str, default='')
parser.add_argument('--output_vocab_path', type=str, default='')
parser.add_argument('--expand_vocab', type=int, default=0)
parser.add_argument('--use_iep_pg', type=int, default=0)
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'])


SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}

def tokenize(s, delim=' ',
            add_start_token=True, add_end_token=True,
            punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    s = s.strip()
    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def build_vocab(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
    token_to_count = {}
    tokenize_kwargs = {
        'delim': delim,
        'punct_to_keep': punct_to_keep,
        'punct_to_remove': punct_to_remove,
    }
    for seq in sequences:
        seq_tokens = tokenize(seq, **tokenize_kwargs,
                              add_start_token=False, add_end_token=False)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count and token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def to_iep_pg(pg):
    iep_pg = []
    for m in pg:
        if m in ['filter_color', 'filter_material', 'filter_shape', 'filter_order']:
            iep_pg[-1] = '_'.join([m, iep_pg[-1]])
        elif m.endswith('events'):
            iep_pg += ['scene', 'get_%s' % m]
        elif m == 'objects':
            iep_pg += ['scene', 'get_objects']
        elif m == 'null':
            iep_pg.append('scene')
        else:
            iep_pg.append(m)
    return iep_pg


def process_mc(q_dict):
    """join the question and choices"""
    qs, pgs, ans = [], [], []
    for cc in q_dict['Correct']:
        qs.append(' '.join([q_dict['question'], cc[0]]))
        pgs.append(cc[1] + q_dict['program'])
        ans.append('yes')
    for cc in q_dict['Wrong']:
        qs.append(' '.join([q_dict['question'], cc[0]]))
        pgs.append(cc[1] + q_dict['program'])
        ans.append('no')
    return qs, pgs, ans


def process_choices(q_dict):
    chs, ch_pgs, ans = [], [], []
    for cc in q_dict['correct']:
        chs.append(cc[0])
        ch_pgs.append(cc[1])
        ans.append('yes')
    for wc in q_dict['wrong']:
        chs.append(wc[0])
        ch_pgs.append(wc[1])
        ans.append('no')
    return chs, ch_pgs, ans


def main(args):
    print('loading questions')
    with open(args.question_path) as f:
        ann = json.load(f)
    all_qs, all_pgs, all_ans = [], [], []
    scene_idxs, q_idxs = [], []
    if args.question_type=='descriptive':
        set_len = len(ann)
        for i in range(set_len):
            q = ann[i]
            all_qs.append(q['question'])
            all_pgs.append(q['program'])
            all_ans.append(q['answer'])
            scene_idxs.append(q['video_id'])
            q_idxs.append(i)
        
#     if args.question_type=='retrieval':
#         set_len = len(ann['expressions'])
#         for i in range(set_len):
#             q = ann['expressions'][i]
#             all_qs.append(q['question'])
#             all_pgs.append(q['program'])
#             all_ans.append(q['answer'])
#             scene_idxs.append(i)
#             q_idxs.append(i)

    else:
        raise ValueError('Invalid question type')
    for i, pg in enumerate(all_pgs):
        all_pgs[i] = ['<START>'] + pg + ['<END>']
    if args.use_iep_pg:
        all_pgs = [to_iep_pg(pg) for pg in all_pgs]
    all_pgs_str = [' '.join(pg) for pg in all_pgs]

    # Create vocabulary
    if args.input_vocab_path == '' or args.expand_vocab == 1:
        print('building question vocab')
        question_token_to_idx = build_vocab(all_qs, punct_to_keep=['\'', ',', '?', '.'])
        print('building program vocab')
        program_token_to_idx = build_vocab(all_pgs_str)
        #print('building answer vocab')
        #answer_token_to_idx = build_vocab(all_ans)
        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'program_token_to_idx': program_token_to_idx,
            #'answer_token_to_idx': answer_token_to_idx,
        }
        print('saving vocab to %s' % args.output_vocab_path)
        
    # Expand vocabulary
    if args.input_vocab_path != '':
        if args.expand_vocab == 1:
            new_vocab = vocab
        with open(args.input_vocab_path) as f:
            vocab = json.load(f)
        if args.expand_vocab == 1:
            num_new_words = 0
            for word in new_vocab['question_token_to_idx']:
                if word not in vocab['question_token_to_idx']:
                    print('Found new word %s' % word)
                    idx = len(vocab['question_token_to_idx'])
                    vocab['question_token_to_idx'][word] = idx
                    num_new_words += 1
            print('Found %d new words' % num_new_words)

    # Output vocabulary
    if args.output_vocab_path != '':
        with open(args.output_vocab_path, 'w') as fout:
            json.dump(vocab, fout)
    # Encode question, program, answer tokens
    print('encoding data')
    qs_encoded = []
    pgs_encoded = []
    ans_encoded = []
    orig_idxs = []
    for i, q in enumerate(all_qs):
        orig_idxs.append(i)
        q_tokens = tokenize(q, punct_to_keep=['\'', ',', '?', '.'])
        q_encoded = encode(q_tokens, vocab['question_token_to_idx'])
        qs_encoded.append(q_encoded)
        pg_encoded = encode(all_pgs[i], vocab['program_token_to_idx'])
        pgs_encoded.append(pg_encoded)

    #pdb.set_trace()
    # Pad encoded sequences
    max_question_length = max(len(x) for x in qs_encoded)
    for qe in qs_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])
    max_program_length = max(len(x) for x in pgs_encoded)
    for pe in pgs_encoded:
        while len(pe) < max_program_length:
            pe.append(vocab['program_token_to_idx']['<NULL>'])

    # Create h5 file
    print('writing output to %s' % args.output_h5_path)
    qs_encoded = np.asarray(qs_encoded, dtype=np.int32)
    pgs_encoded = np.asarray(pgs_encoded, dtype=np.int32)
    print(qs_encoded.shape)
    print(pgs_encoded.shape)
    with h5py.File(args.output_h5_path, 'w') as fout:
        fout.create_dataset('questions', data=qs_encoded)
        fout.create_dataset('programs', data=pgs_encoded)
        fout.create_dataset('video_ids', data=np.asarray(scene_idxs))
        fout.create_dataset('question_idxs', data=np.asarray(q_idxs))
        fout.create_dataset('orig_idxs', data=np.asarray(orig_idxs))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
