class Word2Id():
    def __init__(self, filepath='./word2vec/data/word_to_id.txt'):
        print('Read word_to_id file')
        word_dic = {}
        with open(filepath) as f:
            i = 1
            for line in f:
                word = line.strip()
                word_dic[word] = i
                i += 1
        self.dic = word_dic

    def get_id(self, word):
        try:
            word_id = self.dic[word]
        except KeyError, e:
            word_id = 0
        return word_id

    
def test():
    '''
    Example of Word2Id class usage
    '''
    w = Word2Id()
    print(w.get_id('more'))
    print(w.get_id('dgasdajksfgkjga'))
