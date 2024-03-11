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
    Converts a Unicode string to ASCII. All characters that are not in the first 128 Unicode points
    (i.e., the ASCII range) are removed, focusing on removing diacritics.

    Parameters
    ----------
    s : str
        A Unicode string.

    Returns
    -------
    str
        An ASCII representation of the input string with diacritics removed.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """
    Normalizes a string by converting to ASCII, making lowercase, trimming leading and trailing whitespace,
    replacing ellipses with a period, spacing out punctuation, removing non-letter characters except
    basic punctuation, and replacing multiple spaces with a single space.

    Parameters
    ----------
    s : str
        The string to normalize.

    Returns
    -------
    str
        The normalized string.
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
    """
    Extracts pairs of sentences from a conversational corpus, where each pair consists of an input
    and a response. Filters out pairs with empty sentences.

    Parameters
    ----------
    corpus : object
        A conversational corpus object with methods to get conversation IDs and utterances.

    Returns
    -------
    list of list of str
        A list containing pairs of sentences.
    """
    qa_pairs = []
    for conv_id in corpus.get_conversation_ids():
        conversation = corpus.get_conversation(conv_id)
        utterances = conversation.get_utterance_ids()

        # Iterate over the conversation and extract pairs
        for i in range(len(utterances) - 1):
            input_line = corpus.get_utterance(utterances[i]).text.strip()
            target_line = corpus.get_utterance(utterances[i+1]).text.strip()

            # Filter out pairs with empty sentences
            if input_line and target_line:
                qa_pairs.append([normalizeString(input_line), normalizeString(target_line)])
    return qa_pairs

def loadPreparedData(preprocessed_file):
    """
    Loads sentence pairs from a preprocessed file where each line contains a pair of sentences
    separated by a tab character.

    Parameters
    ----------
    preprocessed_file : str
        The path to the preprocessed file.

    Returns
    -------
    list of list of str
        A list containing pairs of sentences.
    """
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
    """
    Adds all sentences in a list of pairs to a vocabulary object.

    Parameters
    ----------
    voc : Voc
        A vocabulary object.
    pairs : list of list of str
        A list containing pairs of sentences.
    """
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])

def split_data(pairs, split_fraction=0.9):
    """
    Splits a list of pairs into training and validation sets based on a split fraction.

    Parameters
    ----------
    pairs : list of list of str
        A list containing pairs of sentences.
    split_fraction : float, optional
        The fraction of pairs to use for training (default is 0.9).

    Returns
    -------
    tuple of list of list of str
        A tuple containing the training and validation sets.
    """
    train_size = int(len(pairs) * split_fraction)
    
    # Randomly shuffle the data
    random.shuffle(pairs)
    
    # Split the data
    training_pairs = pairs[:train_size]
    validation_pairs = pairs[train_size:]
    
    return training_pairs, validation_pairs

def trimRareWords(voc, pairs, min_count=5):
    """
    Trims rare words from the vocabulary and filters out sentence pairs that contain these words.

    Parameters
    ----------
    voc : Voc
        A vocabulary object.
    pairs : list of list of str
        A list containing pairs of sentences.
    min_count : int, optional
        Words appearing fewer times than this number will be trimmed (default is 5).

    Returns
    -------
    list of list of str
        The filtered list of sentence pairs.
    """
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
    """
    Converts a sentence into a list of word indices according to a given vocabulary. An EOS token
    is appended to the end.

    Parameters
    ----------
    voc : Voc
        A vocabulary object.
    sentence : str
        The sentence to convert.

    Returns
    -------
    list of int
        The list of word indices representing the sentence.
    """
    return [voc.word2index[word] for word in sentence.split(' ')] + [voc.EOS_token]

def zeroPadding(l, fillvalue=0):
    """
    Pads a batch of sentences to the same length with a given fill value.

    Parameters
    ----------
    l : list of list of int
        The batch of sentences represented as word indices.
    fillvalue : int, optional
        The value to pad the sentences with (default is 0).

    Returns
    -------
    list of list of int
        The padded batch of sentences.
    """
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=0):
    """
    Creates a binary mask for the sentences, where 0 indicates padding and 1 indicates a non-padding
    word.

    Parameters
    ----------
    l : list of list of int
        The batch of sentences represented as word indices.
    value : int, optional
        The padding value to check against (default is 0).

    Returns
    -------
    list of list of int
        The binary mask for the batch of sentences.
    """
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
    """
    Prepares the input batch for training by converting sentences to tensors and padding them.

    Parameters
    ----------
    l : list of str
        The list of input sentences.
    voc : Voc
        A vocabulary object.

    Returns
    -------
    tuple
        A tuple containing the padded input tensor and the lengths of sentences before padding.
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc):
    """
    Prepares the output batch for training by converting sentences to tensors, creating a mask, and
    determining the maximum target length.

    Parameters
    ----------
    l : list of str
        The list of output sentences.
    voc : Voc
        A vocabulary object.

    Returns
    -------
    tuple
        A tuple containing the padded output tensor, the mask tensor, and the maximum target length.
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2TrainData(voc, pair_batch):
    """
    Converts a batch of pairs into the format required for training, including input and output
    variables, lengths, masks, and maximum target length.

    Parameters
    ----------
    voc : Voc
        A vocabulary object.
    pair_batch : list of tuples of str
        The batch of sentence pairs.

    Returns
    -------
    tuple
        The training data prepared from the batch, including input and output variables, lengths,
        masks, and the maximum target length.
    """
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

def maskNLLLoss(inp, target, mask):
    """
    Calculates the negative log likelihood loss while ignoring the padding of the output tensor.

    Parameters
    ----------
    inp : Tensor
        The predicted output tensor.
    target : Tensor
        The target tensor.
    mask : Tensor
        The mask tensor indicating where the target is not padding.

    Returns
    -------
    tuple
        The loss and the number of non-pad tokens in the target.
    """
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, teacher_forcing_ratio):
    """
    Performs a single training step including forward and backward passes and parameters update.

    Parameters
    ----------
    input_variable : Tensor
        The batch of input sentences, converted to the appropriate tensor format.
    lengths : Tensor
        The lengths of the sentences in `input_variable`, before padding.
    target_variable : Tensor
        The target sentences tensor.
    mask : Tensor
        The mask tensor for the target sentences.
    max_target_len : int
        The maximum length of the target sentences in the batch.
    encoder : nn.Module
        The encoder model.
    decoder : nn.Module
        The decoder model.
    encoder_optimizer : optim.Optimizer
        The optimizer for the encoder.
    decoder_optimizer : optim.Optimizer
        The optimizer for the decoder.
    batch_size : int
        The batch size.
    clip : float
        The gradient clipping threshold.
    teacher_forcing_ratio : float
        The probability to use teacher forcing during training.

    Returns
    -------
    tuple
        The average loss per non-pad token and the number of non-pad tokens.
    """
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, None)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])

    decoder_hidden = encoder_hidden[:decoder.n_layers]
    
    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_len):

            # Forward pass through decoder
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

            # Forward pass through decoder
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
    """
    Validates the model performance on a given set of validation pairs.

    Parameters
    ----------
    encoder : nn.Module
        The encoder model.
    decoder : nn.Module
        The decoder model.
    searcher : GreedySearchDecoder or another search module
        The decoder search algorithm.
    voc : Voc
        The vocabulary object.
    validation_pairs : list of tuples
        The list of validation sentence pairs.
    batch_size : int
        The batch size.
    max_length : int, optional
        The maximum sentence length (default is set to a global MAX_LENGTH value).

    Returns
    -------
    float
        The average validation loss per non-pad token.
    """
    # Turn off gradients for validation
    with torch.no_grad():
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths, None)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        # Prepare input and output variables
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

        return sum(print_losses), n_totals
    
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=10):
    """
    Evaluates a given sentence with the model using a searcher object.

    Parameters
    ----------
    encoder : nn.Module
        The encoder model.
    decoder : nn.Module
        The decoder model.
    searcher : GreedySearchDecoder or another search module
        The module used for decoding a sequence.
    voc : Voc
        The vocabulary object.
    sentence : str
        The input sentence to evaluate.
    max_length : int, optional
        The maximum length of the output sentence (default is 10).

    Returns
    -------
    list of str
        The decoded output sentence as a list of words.
    """
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]

    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length, SOS_token)
    decoded_words = [voc.index2word[token.item()] for token in tokens]

    # Remove EOS token from end of sentence
    for i in range(len(decoded_words)):
        if decoded_words[i] == 'EOS':
            decoded_words = decoded_words[:i]
            break
    return decoded_words

def calculate_distinct_n_grams(sentences, n=1):
    """
    Calculates the distinct-N metric for a set of sentences, which measures diversity by the ratio of unique N-grams
    over total N-grams in the sentences.

    Parameters
    ----------
    sentences : list of str
        The sentences to calculate the metric for.
    n : int, optional
        The order of N-grams (default is 1).

    Returns
    -------
    float
        The distinct-N metric value.
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

def calculate_f1(predicted, target):
    """
    Calculates the F1 score between a predicted sentence and a target sentence.

    Parameters
    ----------
    predicted : str
        The predicted sentence.
    target : str
        The target sentence.

    Returns
    -------
    float
        The F1 score.
    """
    # Tokenize the sentences if not already tokenized
    predicted_tokens = predicted.split() if isinstance(predicted, str) else predicted
    target_tokens = target.split() if isinstance(target, str) else target
    
    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = sum(pred == targ for pred, targ in zip(predicted_tokens, target_tokens))
    FP = max(len(predicted_tokens) - TP, 0)
    FN = max(len(target_tokens) - TP, 0)
    
    # Calculate Precision and Recall
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    
    # Calculate F1 Score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1