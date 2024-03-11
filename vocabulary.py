class Voc:
    """
    A class to manage the vocabulary for a language for word-level models.

    Parameters
    ----------
    name : str
        The name of the language the vocabulary belongs to.
    PAD_token : int, optional
        The index used for padding (default is 0).
    SOS_token : int, optional
        The index of the Start Of Sentence token (default is 1).
    EOS_token : int, optional
        The index of the End Of Sentence token (default is 2).

    Attributes
    ----------
    name : str
        The language name.
    trimmed : bool
        Flag to indicate if the vocabulary has been trimmed.
    word2index : dict
        A mapping from words to their indices.
    word2count : dict
        A mapping from words to their occurrence counts.
    index2word : dict
        A mapping from indices to their words.
    num_words : int
        The number of unique words in the vocabulary.
    PAD_token : int
        The index used for padding.
    SOS_token : int
        The index for the start-of-sentence token.
    EOS_token : int
        The index for the end-of-sentence token.
    """

    def __init__(self, name, PAD_token=0, SOS_token=1, EOS_token=2):
        self.name = name
        self.trimmed = False 
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count PAD, SOS, EOS
        self.PAD_token = PAD_token
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token

    def addWord(self, word):
        """
        Adds a word to the vocabulary.

        Parameters
        ----------
        word : str
            The word to add to the vocabulary.
        """
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        """
        Adds each word in a sentence to the vocabulary.

        Parameters
        ----------
        sentence : str
            The sentence whose words will be added.
        """
        for word in sentence.split(' '):
            self.addWord(word)

    def trim(self, min_count):
        """
        Trims words below a certain count threshold.

        Parameters
        ----------
        min_count : int
            Words appearing fewer times than this number will be trimmed.
        """
        if self.trimmed:
            return
        self.trimmed = True
        
        keep_words = []
        
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        
        print(f'keep_words {len(keep_words)} / {len(self.word2index)} = {len(keep_words) / len(self.word2index):.4f}')
        
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS"}
        self.num_words = 3  # Reset count to include only PAD, SOS, EOS

        for word in keep_words:
            self.addWord(word)


class Voc_char:
    """
    A class to manage the vocabulary for a language for character-level models.

    Parameters
    ----------
    name : str
        The name of the language the vocabulary belongs to.

    Attributes
    ----------
    name : str
        The language name.
    trimmed : bool
        Flag to indicate if the vocabulary has been trimmed.
    char2index : dict
        A mapping from characters to their indices.
    char2count : dict
        A mapping from characters to their occurrence counts.
    index2char : dict
        A mapping from indices to their characters.
    num_chars : int
        The number of unique characters in the vocabulary.
    """

    def __init__(self, name):
        self.name = name
        self.trimmed = False 
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_chars = 3  # Count PAD, SOS, EOS

    def addChar(self, char):
        """
        Adds a character to the vocabulary.

        Parameters
        ----------
        char : str
            The character to add to the vocabulary.
        """
        if char not in self.char2index:
            self.char2index[char] = self.num_chars
            self.char2count[char] = 1
            self.index2char[self.num_chars] = char
            self.num_chars += 1
        else:
            self.char2count[char] += 1

    def addSentence(self, sentence):
        """
        Adds each character in a sentence to the vocabulary.

        Parameters
        ----------
        sentence : str
            The sentence whose characters will be added.
        """
        for char in sentence:
            self.addChar(char)