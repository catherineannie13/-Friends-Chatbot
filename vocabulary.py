class Voc:
    def __init__(self, name, PAD_token=0, SOS_token=1, EOS_token=2):
        self.name = name
        self.trimmed = False 
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def trim(self, min_count):
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
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 3 
        
        for word in keep_words:
            self.addWord(word)

class Voc_char:
    def __init__(self, name):
        self.name = name
        self.trimmed = False 
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_chars = 3

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.num_chars
            self.char2count[char] = 1
            self.index2char[self.num_chars] = char
            self.num_chars += 1
        else:
            self.char2count[char] += 1

    def addSentence(self, sentence):
        for char in sentence:
            self.addChar(char)