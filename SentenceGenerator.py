from inflector import English

'''
Essentially a dummy holder class for generating sentences. This will
get significantly more sophisticated as we move forward
'''
class SentenceGenerator:
    def __init__(self, keyword):
        self.inf = English()
        self.sentence_prefix = "This dataset is about "
        self.keyword = self.inf.pluralize(keyword)

    def generate(self):
        return self.sentence_prefix + self.keyword