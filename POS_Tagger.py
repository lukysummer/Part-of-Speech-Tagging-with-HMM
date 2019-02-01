import random
from collections import Counter, defaultdict, namedtuple, OrderedDict
from itertools import chain
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution


############################ 1. PROCESS INPUT DATA ############################
Sentence = namedtuple("Sentence", "words tags")

class Dataset(namedtuple("_Dataset", "sentences keys vocab X tagset Y training_set test_set")):
   
    def __new__(cls, tagfile, datafile, train_ratio = 0.8):
        
        ''' 1. READ IN TAGS '''
        with open(tagfile, 'r') as f:
            tags = f.read().split('\n')
        tagset = frozenset(tags)
        
        ''' 2. READ IN INPUT WORDS & CORRESPONDING POS TAGS '''
        with open(datafile, 'r') as f:
            data = [sentence.split('\n') for sentence in f.read().split('\n\n')] 
            
        sentences = OrderedDict((sentence[0],                                 \
                                 Sentence(*zip(*[word.strip().split('\t')     \
                                                 for word in sentence[1:]]    \
                                              )                               \
                                         )                                    \
                                ) for sentence in data if sentence[0]         \
                               )
        
        ''' 3. UNIQUE KEY FOR EACH SENTENCE '''
        keys = tuple(sentences.keys())
        
        ''' 4. CONSTRUCT A VOCABULARY SET OF ALL UNIQUE WORDS '''
        wordset = frozenset(chain(*[s.words for s in sentences.values()]))
        
        ''' 5. WORD AND TAG SEQUENCES AS A TUPLE SET OF TUPLES '''
        word_sequences = tuple([sentences[key].words for key in keys])
        tag_sequences = tuple([sentences[key].tags for key in keys])
        
        ''' 7. SPLIT DATA INTO TRAINING & TEST SETS '''
        key_list = list(keys)
        random.shuffle(key_list)
        split = int(train_ratio * len(key_list))
        training_data = Subset(sentences, key_list[:split])
        test_data = Subset(sentences, key_list[split:])
        
        return super().__new__(cls, 
                               dict(sentences),   # sentences
                               keys,              # keys
                               wordset,           # vocab
                               word_sequences,    # X
                               tagset,            # tagset
                               tag_sequences,     # Y
                               training_data,     # training_set
                               test_data)         # test_set

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())


class Subset(namedtuple("BaseSet", "sentences keys vocab X tagset Y")):
    
    def __new__(cls, sentences, keys):
        
        word_sequences = tuple([sentences[key].words for key in keys])
        tag_sequences = tuple([sentences[key].tags for key in keys])
        wordset = frozenset(chain(*word_sequences))
        tagset = frozenset(chain(*tag_sequences))
        
        return super().__new__(cls, 
                               {key: sentences[key] for key in keys}, 
                               keys, 
                               wordset, 
                               word_sequences,
                               tagset, 
                               tag_sequences)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())
    
    
data = Dataset("tags-universal.txt", "brown-universal.txt")
print("Training set has {} sentences.".format(len(data.training_set.keys)))
print("Test set has {} sentences.\n".format(len(data.test_set.keys)))


################ 2. COUNT NUMBER OF EACH TAG IN ENTIRE CORPUS #################
########################  single_tag_counts[tag] = k  #########################
all_sentences = [list(sentence) for sentence in data.training_set.Y]
all_tags = chain.from_iterable(all_sentences)
single_tag_counts = dict(Counter(all_tags))


############## 3. COUNT NUMBER OF EACH TAG PAIR IN ENTIRE CORPUS ##############
####################  pair_tag_counts[(tag_1, tag_2)] = k  ####################
sentences = [s for s in all_sentences if len(s) > 1]  # discard any sequences of length 1
pairs = []
for s in sentences:
    pairs.extend([(s[i-1], s[i]) for i in range(1, len(s))])

pair_tag_counts = dict(Counter(pairs))

if len(pair_tag_counts) < len(data.training_set.tagset)**2:
    for tag1 in data.training_set.tagset:
        for tag2 in data.training_set.tagset:
            if (tag1, tag2) not in pair_tag_counts:
                pair_tag_counts[(tag1, tag2)] = 0
                

## 4. COUNT NUMBER OF EACH TAG APPEARING IN THE BEGINNING or END OF SENTENCE ##
#########################  start_tag_counts[tag] = k  #########################
##########################  end_tag_counts[tag] = k  ##########################
start_tag_counts = dict(Counter([sentence[0] for sentence in data.training_set.Y]))
end_tag_counts = dict(Counter([sentence[-1] for sentence in data.training_set.Y]))

### if any tag has NO sentences starting/ending with it, set its value to 0:
if len(start_tag_counts) < len(data.training_set.tagset):
    for tag in data.training_set.tagset:
        if tag not in start_tag_counts:
            start_tag_counts[tag] = 0

if len(end_tag_counts) < len(data.training_set.tagset):
    for tag in data.training_set.tagset:
        if tag not in end_tag_counts:
            end_tag_counts[tag] = 0


################## 5. COUNT NUMBER OF (TAG_i, WORD_i) PAIRS ###################
#######################  pair_counts[tag][word] = k  ##########################
pair_counts = defaultdict(lambda: defaultdict(lambda : 0))

for sentence_idx, sentence in enumerate(data.training_set.Y):
    for word_idx, tag in enumerate(sentence):
        word = data.training_set.X[sentence_idx][word_idx]
        pair_counts[tag][word] += 1
        
        
############################# 6. BUILD HMM MODEL ##############################
HMM_model = HiddenMarkovModel(name = "HMM-Tagger")
tag_states = []  # state for each tag

################# (6.1) ADD STATES w/ EMISSION PROBABILITIES ##################
''' 
tag_emissions: P(word_i|tag_j) 
             = P(word_i, tag_j)/P(tag_j)
             = C((word_i, tag_j) pairs)/C(tag_j)
'''
for tag in data.training_set.tagset:
    tag_emissions = DiscreteDistribution({word:pair_counts[tag][word]/single_tag_counts[tag] \
                                          for word in data.training_set.vocab})
    tag_state = State(tag_emissions, name = tag)
    tag_states.append(tag_state)
    HMM_model.add_states(tag_state)

############## (6.2) ADD TRANSITIONS w/ TRANSITION PROBABILITIES ##############
''' 
P(tag_1|START) = P(START, tag_1)/P(START) = C(tag_1 @ start of sentence)/C(START)
                                          = C(tag_1 @ start of sentence)/# of sentences
P(END|tag_1) = P(tag_1, END)/P(tag_1) = C(tag_1 @ end of sentence)/C(tag_1)
P(tag_2|tag_1) = P(tag_1, tag_2)/P(tag_1) = C((word_1, tag_2) pairs)/C(tag_1)
'''
n_sentences = len(data.training_set.keys)

for tag_state1 in tag_states:
    for tag_state2 in tag_states:
        tag1, tag2 = tag_state1.name, tag_state2.name
        HMM_model.add_transition(HMM_model.start, tag_state1, start_tag_counts[tag1]/n_sentences)
        HMM_model.add_transition(tag_state1, HMM_model.end, end_tag_counts[tag1]/single_tag_counts[tag1])
        HMM_model.add_transition(tag_state1, tag_state2, pair_tag_counts[(tag1,tag2)]/single_tag_counts[tag1])
    
HMM_model.bake()    
    
    
#################### 7. MAKE PREDICTIONS ON TRAININING SET ####################
train_correct = 0 # number of correct predictions so far
train_count = 0   # number of predictions so far
print_i = 100

### ITERATE PER SENTENCE
for words, true_tags in zip(data.training_set.X, data.training_set.Y):
    try: 
        # Viterbi Path: most likely sequence of STATES that generated the sequence given  
        _, viterbi_path = HMM_model.viterbi([w for w in words])
        predicted_tags = [state[1].name for state in viterbi_path[1:-1]] 
        train_correct += sum(pred == true for pred, true in zip(predicted_tags, true_tags))
        
        if print_i == 100: # print a sample result
            print("Training Sentence: \n", words)
            print()
            print("Predicted Tags: \n", predicted_tags)
            print()
            print("True Tags: \n", true_tags)
            print_i += 1
    except:
        pass
    train_count += len(words)

train_acc = train_correct/train_count
print("\nTraining Accuracy: {:.2f}%".format(100 * train_acc))
print()
print()
    
    
####################### 8. MAKE PREDICTIONS ON TEST SET #######################
test_correct = 0 
test_count = 0   

### ITERATE PER SENTENCE
for words, true_tags in zip(data.test_set.X, data.test_set.Y):
    try: 
        # Only consider words contained in training set's vocab
        _, viterbi_path = HMM_model.viterbi([w if w in data.training_set.vocab else 'nan' for w in words])
        predicted_tags = [state[1].name for state in viterbi_path[1:-1]] 
        test_correct += sum(pred == true for pred, true in zip(predicted_tags, true_tags))
        
        if print_i == 101: # print a sample result
            print("Test Sentence: \n", words)
            print()
            print("Predicted Tags: \n", predicted_tags)
            print()
            print("True Tags: \n", true_tags)
            print_i += 1
    except:
        pass
    test_count += len(words)

test_acc = test_correct/test_count
print("\nTest Accuracy: {:.2f}%".format(100 * test_acc))