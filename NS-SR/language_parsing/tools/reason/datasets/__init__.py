# from .my_dataset import MyQuestionDataset
from .sr_dataset import SRDataset
from torch.utils.data import DataLoader


def get_dataset(opt, split, run_test=None):
    """Get function for dataset class"""
    assert split in ['train', 'val', 'test', 'grounding']
#     print('dataset opt', opt)
    if opt.dataset == 'sr_dataset':
        if split == 'train':
            question_h5_path = opt.mc_q_train_question_path
            max_sample = opt.max_train_samples
        elif split == 'val':
            question_h5_path = opt.mc_q_val_question_path
            max_sample = opt.max_val_samples
        else:
            question_h5_path = opt.mc_q_test_question_path
            max_sample = None
        vocab_path = opt.mc_q_vocab_path
    elif opt.dataset == 'oe':
        if split == 'train':
            question_h5_path = opt.oe_train_question_path
            max_sample = opt.max_train_samples
        elif split == 'val':
            question_h5_path = opt.oe_val_question_path
            max_sample = opt.max_val_samples
        else:
            question_h5_path = opt.oe_test_question_path
            max_sample = None
        vocab_path = opt.oe_vocab_path
    elif opt.dataset == 'mc_q':
        if split == 'train':
            question_h5_path = opt.mc_q_train_question_path
            max_sample = opt.max_train_samples
        elif split == 'val':
            question_h5_path = opt.mc_q_val_question_path
            max_sample = opt.max_val_samples
        else:
            question_h5_path = opt.mc_q_test_question_path
            max_sample = None
        vocab_path = opt.mc_q_vocab_path
    elif opt.dataset == 'mc_c':
        if split == 'train':
            question_h5_path = opt.mc_c_train_question_path
            max_sample = opt.max_train_samples
        elif split == 'val':
            question_h5_path = opt.mc_c_val_question_path
            max_sample = opt.max_val_samples
        else:
            question_h5_path = opt.mc_c_test_question_path
            max_sample = None
        vocab_path = opt.mc_c_vocab_path
    elif opt.dataset == 'retrieval':
        if split == 'train':
            question_h5_path = opt.retrieval_train_question_path
            max_sample = opt.max_train_samples
        elif split == 'val':
            question_h5_path = opt.retrieval_val_question_path
            max_sample = opt.max_val_samples
        vocab_path = opt.retrieval_vocab_path
    elif opt.dataset == 'grounding':
        if split == 'train':
            question_h5_path = opt.grounding_train_question_path
            max_sample = opt.max_train_samples
        elif split == 'val':
            question_h5_path = opt.grounding_val_question_path
            max_sample = opt.max_val_samples
        vocab_path = opt.retrieval_vocab_path
    else:
        raise ValueError('Invalid dataset')

    if not run_test:
        run_test = (split == 'test')
    dataset = SRDataset(question_h5_path, max_sample, vocab_path, test=run_test)
    return dataset


def get_dataloader(opt, split, run_test=None):
    """Get function for dataloader class"""
    dataset = get_dataset(opt, split, run_test)
    shuffle = (opt.shuffle == 1) and (split == 'train')
    loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.num_workers)
    print('| %s %s loader has %d samples' % (opt.dataset, split, len(loader.dataset)))
    return loader
