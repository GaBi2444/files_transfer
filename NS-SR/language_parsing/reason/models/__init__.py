from .encoder import Encoder
from .decoder import Decoder
from .seq2seq import Seq2seq
import reason.utils.utils as utils


def get_vocab(opt):
    if opt.dataset == 'sr_dataset':
        vocab_json = opt.mc_q_vocab_path
    elif opt.dataset == 'oe':
        vocab_json = opt.oe_vocab_path
    elif opt.dataset == 'mc_c':
        vocab_json = opt.mc_c_vocab_path
    elif opt.dataset == 'mc_q':
        vocab_json = opt.mc_q_vocab_path
    elif opt.dataset == 'retrieval':
        vocab_json = opt.retrieval_vocab_path
    elif opt.dataset == 'grounding':
        vocab_json = opt.grounding_vocab_path
    else:
        raise ValueError('Invalid dataset')
    vocab = utils.load_vocab(vocab_json)
    return vocab


def create_seq2seq_net(input_vocab_size, output_vocab_size, hidden_size, 
                       word_vec_dim, n_layers, bidirectional, variable_lengths, 
                       use_attention, encoder_max_len, decoder_max_len, start_id, 
                       end_id, word2vec_path=None, fix_embedding=False):
    word2vec = None
    if word2vec_path is not None:
        word2vec = utils.load_embedding(word2vec_path)

    encoder = Encoder(input_vocab_size, encoder_max_len, 
                      word_vec_dim, hidden_size, n_layers,
                      bidirectional=bidirectional, variable_lengths=variable_lengths,
                      word2vec=word2vec, fix_embedding=fix_embedding)
    decoder = Decoder(output_vocab_size, decoder_max_len,
                      word_vec_dim, hidden_size, n_layers, start_id, end_id,
                      bidirectional=bidirectional, use_attention=use_attention)

    return Seq2seq(encoder, decoder)
