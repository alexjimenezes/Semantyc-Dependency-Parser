import numpy as np
import pickle
import random
from tqdm import tqdm

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''             ADDS FEATURES TO THE FEATURE MAP               '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class FeatureMapper:
    def __init__(self):
        self.feature_map = {}
        '''Tracks the number of features'''
        self.id = 1
        self.frozen = False

    def feature_template(self, state, sentence):
        feature_list = np.empty((0), int)
        stack = state.stack
        buffer = state.buffer
        ld = state.ld
        rd = state.rd
        form = sentence.form
        pos = sentence.pos

        if stack:
            s0 = stack[-1]
            ''' First element of stack '''
            feature_list = np.append(feature_list, self.get_feature("s0_form=" + form[s0]))
            feature_list = np.append(feature_list, self.get_feature("s0_pos=" + pos[s0]))
            feature_list = np.append(feature_list, self.get_feature("s0_form_pos=" + form[s0] + pos[s0]))

            s0_heads = []
            for (p, c, r) in state.arc_rels:
                if c == s0:
                    s0_heads.append((p, r))

            ''' Heads of stack[-1] if any '''
            if s0_heads:
                for i in range(0, len(s0_heads) - 1):
                    feature_list = np.append(feature_list,
                                             self.get_feature("head_s0" + str(i) + "_form=" + form[s0_heads[i][0]]))
                    feature_list = np.append(feature_list,
                                             self.get_feature("head_s0" + str(i) + "_pos=" + pos[s0_heads[i][0]]))
                    feature_list = np.append(feature_list,
                                             self.get_feature("head_s0" + str(i) + "_rel=" + str(s0_heads[i][0])))
            ''' Left and Right most dependencies '''
            if ld[s0] >= 0:
                feature_list = np.append(feature_list, self.get_feature("ld_s0_form=" + form[ld[s0]]))
                feature_list = np.append(feature_list, self.get_feature("ld_s0_pos=" + pos[ld[s0]]))
            if rd[s0] >= 0:
                feature_list = np.append(feature_list, self.get_feature("rd_s0_form=" + form[rd[s0]]))
                feature_list = np.append(feature_list, self.get_feature("rd_s0_pos=" + pos[rd[s0]]))

            ''' Second to last element of stack '''
            if len(stack) > 1:
                ''' For stack[1] element '''
                s1 = stack[-2]
                feature_list = np.append(feature_list, self.get_feature("s1_form=" + form[s1]))
                feature_list = np.append(feature_list, self.get_feature("s1_pos=" + pos[s1]))
                feature_list = np.append(feature_list, self.get_feature("s1_form_pos=" + form[s1] + pos[s1]))

        if buffer:
            ''' First element of buffer '''
            b0 = buffer[0]
            feature_list = np.append(feature_list, self.get_feature("b0_form=" + form[b0]))
            feature_list = np.append(feature_list, self.get_feature("b0_pos=" + pos[b0]))
            feature_list = np.append(feature_list, self.get_feature("b0_form_pos=" + form[b0] + pos[b0]))

            ''' heads, ld and rd for buffer[0] element '''
            b0_heads = []
            for (p, c, r) in state.arc_rels:
                if c == b0:
                    b0_heads.append((p, r))

            if b0_heads:
                for i in range(0, len(b0_heads) - 1):
                    feature_list = np.append(feature_list,
                                             self.get_feature("head_b0" + str(i) + "_form=" + form[b0_heads[i][0]]))
                    feature_list = np.append(feature_list,
                                             self.get_feature("head_b0" + str(i) + "_pos=" + pos[b0_heads[i][0]]))
                    feature_list = np.append(feature_list,
                                             self.get_feature("head_b0" + str(i) + "_rel=" + str(b0_heads[i][0])))
            if ld[b0] >= 0:
                feature_list = np.append(feature_list, self.get_feature("ld_b0_form=" + form[ld[b0]]))
                feature_list = np.append(feature_list, self.get_feature("ld_b0_pos=" + pos[ld[b0]]))
            if rd[b0] >= 0:
                feature_list = np.append(feature_list, self.get_feature("rd_b0_form=" + form[rd[b0]]))
                feature_list = np.append(feature_list, self.get_feature("rd_b0_pos=" + pos[rd[b0]]))

            ''' Second element of buffer '''
            if len(buffer) > 1:
                b1 = buffer[1]
                feature_list = np.append(feature_list, self.get_feature("b1_form=" + form[b1]))
                feature_list = np.append(feature_list, self.get_feature("b1_pos=" + pos[b1]))
                feature_list = np.append(feature_list, self.get_feature("b1_form_pos=" + form[b1] + pos[b1]))
                feature_list = np.append(feature_list, self.get_feature("b0_form+b1_form=" + form[b0] + form[b1]))
                feature_list = np.append(feature_list, self.get_feature("b0_pos+b1_pos=" + pos[b0] + pos[b1]))
                if stack:
                    feature_list = np.append(feature_list, self.get_feature(
                        "b0_pos+b1_pos+s0_pos=" + pos[b0] + pos[b1] + pos[stack[-1]]))

            ''' Third element of buffer '''
            if len(buffer) > 2:
                b2 = buffer[2]
                feature_list = np.append(feature_list, self.get_feature("b2_pos=" + pos[b2]))
                feature_list = np.append(feature_list, self.get_feature("b2_form=" + form[b2]))
                feature_list = np.append(feature_list, self.get_feature("b2_form_pos=" + form[b2] + pos[b2]))
                feature_list = np.append(feature_list,
                                         self.get_feature("b0_pos+b1_pos+b2_pos=" + pos[b0] + pos[b1] + pos[b2]))
            if len(buffer) > 3:
                ''' buffer[3] POS only '''
                b3 = buffer[3]
                feature_list = np.append(feature_list, self.get_feature("b3_pos=" + pos[b3]))

        if stack and buffer:
            s0 = stack[-1]
            b0 = buffer[0]
            ''' Both stack and buffer '''
            feature_list = np.append(feature_list, self.get_feature(
                "s0_form_pos+b0_form_pos=" + form[s0] + pos[s0] + form[b0] + pos[b0]))
            feature_list = np.append(feature_list,
                                     self.get_feature("s0_form_pos+b0_form=" + form[s0] + pos[s0] + form[b0]))
            feature_list = np.append(feature_list,
                                     self.get_feature("s0_form+b0_form_pos=" + form[s0] + form[b0] + pos[b0]))
            feature_list = np.append(feature_list,
                                     self.get_feature("s0_form_pos+b0_pos=" + form[s0] + pos[s0] + pos[b0]))
            feature_list = np.append(feature_list,
                                     self.get_feature("s0_pos+b0_form_pos=" + pos[s0] + form[b0] + pos[b0]))
            feature_list = np.append(feature_list, self.get_feature("s0_form+b0_form=" + form[s0] + form[b0]))
            feature_list = np.append(feature_list, self.get_feature("s0_pos+b0_pos=" + pos[s0] + pos[b0]))

            ''' Distance between stack[0] and buffer[0] '''
            distance = str(b0 - s0)

            if ld[b0] >= 0:
                feature_list = np.append(feature_list,
                                         self.get_feature("s0_pos+b0_pos+ld_s0_pos=" + pos[s0] + pos[b0] + pos[ld[s0]]))
                feature_list = np.append(feature_list,
                                         self.get_feature("s0_pos+b0_pos+ld_b0_pos=" + pos[s0] + pos[b0] + pos[ld[b0]]))
            if rd[b0] >= 0:
                feature_list = np.append(feature_list,
                                         self.get_feature("s0_pos+b0_pos+rd_s0_pos=" + pos[s0] + pos[b0] + pos[rd[s0]]))
                feature_list = np.append(feature_list,
                                         self.get_feature("s0_pos+b0_pos+rd_b0_pos=" + pos[s0] + pos[b0] + pos[rd[b0]]))

            ''' combined distance of s0 and b0 features'''
            feature_list = np.append(feature_list, self.get_feature("s0_form+dist=" + form[s0] + distance))
            feature_list = np.append(feature_list, self.get_feature("s0_pos+dist=" + pos[s0] + distance))
            feature_list = np.append(feature_list, self.get_feature("b0_form+dist=" + form[b0] + distance))
            feature_list = np.append(feature_list, self.get_feature("b0_pos+dist=" + pos[b0] + distance))
            feature_list = np.append(feature_list,
                                     self.get_feature("s0_form+b0_form+dist=" + form[s0] + form[b0] + distance))
            feature_list = np.append(feature_list,
                                     self.get_feature("s0_pos+b0_pos+dist=" + pos[s0] + pos[b0] + distance))

        return feature_list

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''          CHECKS MAP AND ADDS IF IT DOES NOT EXIST          '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def get_feature(self, feature):
        if self.frozen:
            if feature not in self.feature_map:
                return 0
            else:
                return self.feature_map[feature]
        else:
            if feature not in self.feature_map:
                self.feature_map[feature] = self.id
                self.id += 1
            return self.feature_map[feature]


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''    MODEL TRAINED TO BE USED IN THE TEST AND DEV MODES      '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class Model:
    def __init__(self, feature_map, weights, total_labels):
        pass
        self.map = feature_map
        self.weights = weights
        self.total_labels = total_labels

    def save_model(self, model):
        f = open('model', 'wb')
        pickle.dump(model, f, -1)
        f.close()

    ''' Train running an averaged perceptron'''
    def train(self, train_data):
        print("Training started...")
        u = np.zeros(self.weights.shape, dtype=np.float32)
        q = 0
        n_labels = self.weights.shape[0]

        for epoch in range(0, 15):
            print("\nRunning epoch " + str(epoch + 1) + "of 15")
            correct = 0
            j = 0
            random.shuffle(train_data, random.random)
            for data in tqdm(train_data):
                q += 1
                j += 1
                scores = np.zeros((n_labels,))
                feature_vector = np.ones((len(data.feature_vector),), dtype=np.float32)

                for index in data.feature_vector:
                    for i in range(0, n_labels):
                        scores[i] += self.weights[i][index]
                predicted = self.total_labels[np.argmax(scores)]

                if predicted != data.label:
                    for index in data.feature_vector:
                        self.weights[self.total_labels.index(data.label)][index] += 1
                        self.weights[self.total_labels.index(predicted)][index] -= 1
                        u[self.total_labels.index(data.label)][index] += q
                        u[self.total_labels.index(predicted)][index] -= q

                if predicted == data.label:
                    correct += 1

                if j % 50000 == 0:
                    print("States", j, ": ", (correct / j))
            print("Accuracy: ", (correct / len(train_data)))
        self.weights -= u * (1 / q)
