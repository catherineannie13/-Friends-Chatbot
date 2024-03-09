import random
import itertools
import torch
import unicodedata
import re
PAD_token = 0
SOS_token = 1
EOS_token = 2

def unicodeToAscii(s):
    """
    Converts the Unicode string to plain ASCII.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """
    Lowercase, trim, replace ellipses with a single full stop, 
    and remove non-letter characters except for basic punctuation.
    """
    # Convert to ASCII
    s = unicodeToAscii(s.lower().strip()) 
    # Replace ellipses with a single full stop
    s = re.sub(r"\.\s*\.\s*\.\s*", ". ", s)
    # Space out punctuation
    s = re.sub(r"([.!?,])", r" \1", s)
    # Remove non-letter characters except for basic punctuation
    s = re.sub(r"[^a-zA-Z.!?,']+", r" ", s)
    # Replace multiple spaces with a single space
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def extractSentencePairs(corpus):
    qa_pairs = []
    for conv_id in corpus.get_conversation_ids():
        conversation = corpus.get_conversation(conv_id)
        utterances = conversation.get_utterance_ids()
        for i in range(len(utterances) - 1):
            input_line = corpus.get_utterance(utterances[i]).text.strip()
            target_line = corpus.get_utterance(utterances[i+1]).text.strip()
            if input_line and target_line:  # Filter out empty lines
                qa_pairs.append([normalizeString(input_line), normalizeString(target_line)])
    return qa_pairs

def loadPreparedData(preprocessed_file):
    pairs = []
    with open(preprocessed_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                pairs.append(parts)
            else:
                print(f"Skipping malformed line: {line.strip()}")
    return pairs

def addPairsToVoc(voc, pairs):
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])

def split_data(pairs, split_fraction=0.9):
    # Calculate the size of the training set
    train_size = int(len(pairs) * split_fraction)
    
    # Randomly shuffle the data
    random.shuffle(pairs)
    
    # Split the data
    training_pairs = pairs[:train_size]
    validation_pairs = pairs[train_size:]
    
    return training_pairs, validation_pairs

def trimRareWords(voc, pairs, min_count=5):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(min_count)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

def indexesFromSentence(voc, sentence):
    return [voc.word2index.get(word) for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=0):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, teacher_forcing_ratio):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, None)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_input = target_variable[t].view(1, -1)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1)

            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()

    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses), n_totals

def validate(encoder, decoder, batch_size, input_variable, lengths, target_variable, mask, max_target_len):
    with torch.no_grad():  # No gradients needed for validation
        loss = 0
        print_losses = []
        n_totals = 0

        encoder_outputs, encoder_hidden = encoder(input_variable, lengths, None)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

        return sum(print_losses) / n_totals
    
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=10):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def calculate_distinct_n_grams(sentences, n=1):
    """
    Calculate Distinct-N metric for a list of sentences.
    
    Parameters:
    - sentences: list of strings, where each string is a generated response or sentence.
    - n: integer, the n-gram length (1 for unigrams, 2 for bigrams, etc.)
    
    Returns:
    - distinct_n: The distinct-N metric, calculated as the number of unique n-grams
                  divided by the total number of n-grams.
    """
    # Tokenize the sentences into lists of words
    tokens_list = [sentence.split() for sentence in sentences]
    
    # Generate n-grams from tokens
    n_grams = list(itertools.chain(*[list(zip(*[tokens[i:] for i in range(n)])) for tokens in tokens_list]))
    
    # Count unique n-grams and total n-grams
    unique_n_grams = len(set(n_grams))
    total_n_grams = len(n_grams)
    
    # Calculate distinct-N metric
    distinct_n = unique_n_grams / total_n_grams if total_n_grams > 0 else 0
    return distinct_n