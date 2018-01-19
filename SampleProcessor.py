import random
import numpy as np

class SampleProcessor:
    def __init__(self, similarity_func=None, classes=None, vprint=print):
        self.vprint = vprint
        self.sim_scores = {}
        self.n_samples_seen = {}
        self.classes=classes
        self.similarity_func = similarity_func

    def process_data(self, data, max_num_samples):
        for source, text in data.items():
            self.process_samples(source, text, max_num_samples)

        return self.sim_scores, self.n_samples_seen

    def process_samples(self, source, text, max_num_samples):

        self.vprint('processing samples from:', source)
        
        if max_num_samples and len(text) > max_num_samples:
            self.vprint('subsampling word list of length {0} to {1}'.format(len(text), max_num_samples))
            # shuffle(text)  # TODO problem to shuffle in place -- effects outside method? 
            text = random.sample(text, max_num_samples) if max_num_samples < len(text) else text

        for words in text:  # TODO vectorize
            try:
                if not source in self.sim_scores.keys():
                    self.sim_scores[source] = np.zeros(len(self.classes))
                
                if not source in self.n_samples_seen.keys():
                    self.n_samples_seen[source] = 0

                self.sim_scores[source] += self.similarity_func(words)
                self.n_samples_seen[source] += 1

            except KeyError as err:
                print('error checking distance of word {0} to classes (out of vocab?):'.format(words), err)
                raise err
            except Exception as err:
                print('unknown error: ', err)
                print('text being processed: {0}'.format(words))
                raise err

    def reset_scores(self):
        # dictionaries with data source as keys 
        self.n_samples_seen = {}
        self.sim_scores = {}