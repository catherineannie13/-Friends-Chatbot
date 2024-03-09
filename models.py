import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Ensure n_layers is an integer
        n_layers = int(n_layers)
        
        self.embedding_dropout = nn.Dropout(dropout)
        # Ensure dropout is 0.0 if n_layers == 1
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0.0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden, attn_weights
    
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length, SOS_token):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], dtype=torch.long)
        all_scores = torch.zeros([0])
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
    
class BeamSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, voc, beam_width=5):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.voc = voc
        self.beam_width = beam_width

    def forward(self, input_seq, input_length, max_length, SOS_token):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length, None)

        # Tensor to store decoder outputs
        all_tokens = torch.zeros([0], dtype=torch.long)
        all_scores = torch.zeros([0])

        # Start with the start of the sentence token
        decoder_input = torch.ones(1, 1, dtype=torch.long) * SOS_token
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Now, instead of just looking for the most likely next step, 
        # keep track of the top k choices at each step; initialize beams
        beams = [(decoder_input, decoder_hidden, 0, [])] # Each beam is a tuple (decoder_input, decoder_hidden, score, token_list)

        for _ in range(max_length):
            candidates = []
            for beam in beams:
                decoder_input, decoder_hidden, score, token_list = beam
                with torch.no_grad():
                    decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                log_probs = F.log_softmax(decoder_output, dim=1)
                top_log_probs, top_idxs = log_probs.topk(self.beam_width)

                for i in range(self.beam_width):
                    next_token_log_prob = top_log_probs[0][i].item()
                    next_token_idx = top_idxs[0][i].item()

                    candidates.append((torch.ones(1, 1, dtype=torch.long) * next_token_idx, decoder_hidden, score + next_token_log_prob, token_list + [next_token_idx]))

            # Sort candidates by score and select top k
            candidates = sorted(candidates, key=lambda x: x[2], reverse=True)[:self.beam_width]
            beams = candidates

        # Choose the beam with the highest score
        _, _, final_score, final_tokens = max(beams, key=lambda x: x[2])

        all_tokens = torch.tensor(final_tokens)
        all_scores = torch.tensor(final_score)
        return all_tokens, all_scores