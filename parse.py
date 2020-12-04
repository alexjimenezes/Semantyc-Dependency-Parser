import pickle
from _collections import deque

from model import FeatureMapper, Model
from reader import FileDecoder
from writer import FileEncoder
import numpy as np
from tqdm import tqdm

LA = 0
RA = 1
RD = 2
SH = 3

ROOT = 0

HEAD = 0
CHILD = 1
REL = 2


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''       CONTAINS THE STATE AT ANY MOMENT OF A SENTENCE       '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class State:
    def __init__(self, stack, buffer):
        self.stack = deque(stack)
        self.buffer = deque(buffer)
        '''all yje arcs found in the sentence'''
        self.arc_rels = []
        '''left most and right most dependency of the given token'''
        self.ld = np.empty(len(self.buffer) + len(self.stack), dtype=np.int32)
        self.rd = np.empty(len(self.buffer) + len(self.stack), dtype=np.int32)
        self.ld.fill(-1)
        self.rd.fill(-1)

    def get_dependency(self, head):
        dep_list = []
        for i in range(0, len(self.arc_rels)):
            if self.arc_rels[i][HEAD] == head:
                dep_list.append(self.arc_rels[i][CHILD])
        return dep_list


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' CONTAINS THE ACTION AND THE FEATURES PRESENT AT THAT STATE '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class Instance:
    def __init__(self, label, feature_vector):
        self.label = label
        self.feature_vector = feature_vector


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''   PARSER CLASS WILL PARSE BOTH FOR TRAINING AND TESTING    '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class Parser:
    def __init__(self, state, sentence, feature_map):
        self.sentence = sentence
        self.state = state
        self.map = feature_map
        self.data = []

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''         DECIDE ACTION BASED ON DAG SENTENCES  INPUT        '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def oracle(self):
        train_data = []
        '''Take actions until buffer is empty'''
        while self.state.buffer:
            feature_list = np.asarray(self.map.feature_template(self.state, self.sentence))
            if self.state.stack and self.should_left_arc():
                self.left_arc(train=True)
                instance = Instance(str(LA) + "_" + self.state.arc_rels[-1][REL], feature_list)
            elif self.state.stack and self.should_right_arc():
                self.right_arc(train=True)
                instance = Instance(str(RA) + "_" + self.state.arc_rels[-1][REL], feature_list)
            elif self.state.stack and self.should_reduce():
                self.reduce()
                instance = Instance(str(RD), feature_list)
            else:
                self.shift()
                instance = Instance(str(SH), feature_list)
            train_data.append(instance)
        self.data = train_data
        return self.data

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''         PARSE SENTENCES BASED ON THE TRAINING MODEL        '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def parse(self, loaded_model):
        weights = loaded_model.weights
        '''Parse until buffer is empty'''
        while self.state.buffer:
            feature_list = self.map.feature_template(self.state, self.sentence)
            scores = np.zeros((len(loaded_model.total_labels),))

            '''Update the scores array using the trained model'''
            for index in feature_list:
                for i in range(0, len(loaded_model.total_labels)):
                    scores[i] += weights[i][index]

            '''Sort the weights giving an array of the indexes'''
            predicted = np.argsort(-scores)
            '''Iterate over the predicted actions until one can be executed'''
            for item in predicted:
                label = int(loaded_model.total_labels[item].split('_', 1)[0])
                if len(loaded_model.total_labels[item].split('_', 1)) > 1:
                    rel = loaded_model.total_labels[item].split('_', 1)[1]
                if label == LA and self.state.stack and self.can_left_arc():
                    self.left_arc(rel=rel)
                    break
                elif label == RA and self.state.stack and self.can_right_arc():
                    self.right_arc(rel=rel)
                    break
                elif label == RD and self.state.stack and self.can_reduce():
                    self.reduce()
                    break
                elif label == SH and self.state.stack:
                    self.shift()
                    break

        return self.state.arc_rels

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ''' ACTIONS THAT THE ORACLE SHOULD TAKE FROM THE TRAINING DATA '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def should_left_arc(self):
        stack_top = self.state.stack[-1]
        buff_front = self.state.buffer[0]
        if buff_front in self.sentence.heads[stack_top]:
            return True
        return False

    def should_right_arc(self):
        stack_top = self.state.stack[-1]
        buff_front = self.state.buffer[0]
        if stack_top in self.sentence.heads[buff_front]:
            return True
        return False

    def should_reduce(self):
        stack_top = self.state.stack[-1]
        for (head, child, rel) in self.sentence.gold_sdp_arcs:
            if head == stack_top or child == stack_top:
                return False
        return True

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''    CHECK IF THE PARSER CAN EXECUTE THE SELECTED ACTION     '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def can_left_arc(self):
        stack_top = self.state.stack[-1]
        buf_front = self.state.buffer[0]
        '''Can make the ROOT dependant'''
        if stack_top == ROOT:
            return False
        '''Check that the same arc does not exist nor the opposite arc'''
        for arc in self.state.arc_rels:
            if (arc[0] == stack_top and arc[1] == buf_front) or (arc[0] == buf_front and arc[1] == stack_top):
                return False
        return True

    def can_right_arc(self):
        stack_top = self.state.stack[-1]
        buf_front = self.state.buffer[0]
        '''Check that the same arc does not exist nor the opposite arc'''
        for arc in self.state.arc_rels:
            if (arc[0] == stack_top and arc[1] == buf_front) or (arc[0] == buf_front and arc[1] == stack_top):
                return False
        return True

    def can_reduce(self):
        if len(self.state.stack) > 2:
            return True
        return False
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''                        TAKE THE ACTION                     '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def left_arc(self, train=False, rel=None):
        stack_top = self.state.stack[-1]
        buff_front = self.state.buffer[0]

        if train:
            '''Remove head for future iterations'''
            self.sentence.heads[stack_top].remove(buff_front)
            '''Remove arc from sentence for future iterations'''
            for (head, child, rel) in self.sentence.gold_sdp_arcs:
                if head == buff_front and child == stack_top:
                    '''Add arc to state arcs'''
                    self.state.arc_rels.append((head, child, rel))
                    self.sentence.gold_sdp_arcs.remove((head, child, rel))
        else:
            self.state.arc_rels.append((buff_front, stack_top, rel))

        '''Get the left most dependency'''
        self.state.ld[self.state.buffer[0]] = min(self.state.get_dependency(self.state.buffer[0]))
        '''Get the right most dependency'''
        self.state.rd[self.state.buffer[0]] = max(self.state.get_dependency(self.state.buffer[0]))

    def right_arc(self, train=False, rel=None):
        stack_top = self.state.stack[-1]
        buff_front = self.state.buffer[0]
        if train:
            '''Remove head for future iterations'''
            self.sentence.heads[buff_front].remove(stack_top)
            '''Remove arc from sentence for future iterations'''
            for (head, child, rel) in self.sentence.gold_sdp_arcs:
                if head == stack_top and child == buff_front:
                    '''Add arc to state arcs'''
                    self.state.arc_rels.append((head, child, rel))
                    self.sentence.gold_sdp_arcs.remove((head, child, rel))
        else:
            self.state.arc_rels.append((stack_top, buff_front, rel))

        self.state.ld[stack_top] = min(self.state.get_dependency(stack_top))
        self.state.rd[stack_top] = max(self.state.get_dependency(stack_top))

    def reduce(self):
        self.state.stack.pop()

    def shift(self):
        last_buffer = self.state.buffer.popleft()
        self.state.stack.append(last_buffer)


if __name__ == "__main__":
    model_type = "train"
    init_stack = np.arange(1)
    feature_map = FeatureMapper()

    '''TRAINING SEQUENCE'''
    if model_type == "train":
        sentences = FileDecoder("train.sdp", "train").sentences
        train_data = []
        total_labels = []
        print("\nThe Oracle is working it's magic ...")
        for s in tqdm(sentences):
            init_buffer = np.arange(1, len(s.form))
            state = State(init_stack, init_buffer)
            parser_data = Parser(state, s, feature_map).oracle()
            train_data.append(parser_data)
        print("The Oracle finished processing sentences!")

        '''Get total possible labels for the model'''
        print("\nExtracting total number of generated labels...")

        for data_set in tqdm(train_data):
            for data in data_set:
                if data.label not in total_labels:
                    total_labels.append(data.label)
        print("There are " + str(len(total_labels)) + " in total!")

        train_data = np.concatenate(train_data).flatten().tolist()
        feature_map.frozen = True
        print("\nCreating vector for weights...")
        weights = np.zeros((len(total_labels), feature_map.id), dtype=np.float32)
        print("Vector has this shape: " + str(weights.shape))
        model = Model(feature_map, weights, total_labels)
        print("\nStarting averaged perceptron algorithm...")
        model.train(train_data)
        print("\nSaving model...")
        model.save_model(model)
        print("\nModel saved!")

    ''' TESTING SEQUENCE '''
    if model_type == "test":
        file1 = "esl.input"
        output1 = "esl.output"
        file2 = "cesl.input"
        output2 = "cesl.output"

        '''Load Model'''
        f = open('model', 'rb')
        loaded_model = pickle.load(f)
        print("Model loaded!")
        f.close()

        sentences = FileDecoder(file2, model_type).sentences
        print("\nParsing sentences...")
        ''' Write arcs to file '''
        for sentence in tqdm(sentences):
            init_buffer = np.arange(1, len(sentence.form))
            state = State(init_stack, init_buffer)
            sentence.arcs = Parser(state, sentence, loaded_model.map).parse(loaded_model)
        print("Done parsing.")
        print("\nWriting to " + output2)
        FileEncoder(sentences, file2, output2)

    ''' DEV MODE GIVES BACK THE SCORES '''
    if model_type == "dev":
        sentences = FileDecoder("dev.sdp", model_type).sentences
        gold_sentences = FileDecoder("dev.sdp", model_type).sentences
        f = open('model', 'rb')
        loaded_model = pickle.load(f)
        print("Model loaded!")
        f.close()
        print("\nParsing sentences...")
        l_true_positive = 0
        l_true_positive_plus_false_positive = 0
        l_true_positive_plus_false_negative = 0

        u_true_positive = 0
        u_true_positive_plus_false_positive = 0
        u_true_positive_plus_false_negative = 0

        for (sentence, gold_sentence) in tqdm(zip(sentences, gold_sentences)):
            init_buffer = np.arange(1, len(sentence.form))
            state = State(init_stack, init_buffer)
            arcs = Parser(state, sentence, loaded_model.map).parse(loaded_model)
            '''Labelled'''
            l_arcs_set = set(arcs)
            l_gold_arcs_set = set(gold_sentence.gold_sdp_arcs)
            l_true_positive += len(l_arcs_set.intersection(l_gold_arcs_set))
            l_true_positive_plus_false_positive += len(l_arcs_set)
            l_true_positive_plus_false_negative += len(l_gold_arcs_set)

            '''Unlabelled'''
            u_arcs_set = set([(item[0], item[1]) for item in l_arcs_set])
            u_gold_arcs_set = set([(item[0], item[1]) for item in l_gold_arcs_set])
            u_true_positive += len(u_arcs_set.intersection(u_gold_arcs_set))
            u_true_positive_plus_false_positive += len(u_arcs_set)
            u_true_positive_plus_false_negative += len(u_gold_arcs_set)
        print("Done parsing.")

        print("\nLabelled Results:")
        lp = l_true_positive / l_true_positive_plus_false_positive
        print("\tLP: ", lp)
        lr = l_true_positive / l_true_positive_plus_false_negative
        print("\tLR: ", lr)
        print("\tLF: ", 2 * lr * lp / (lr + lp))

        print("\nUnlabelled Results:")
        up = u_true_positive / u_true_positive_plus_false_positive
        print("\tUP: ", up)
        ur = u_true_positive / u_true_positive_plus_false_negative
        print("\tUR: ", ur)
        print("\tUF: ", 2 * ur * up / (ur + up))
