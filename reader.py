ROOT_ID = 0
ID = 0
FORM = 1
LEMMA = 2
POS = 3
TOP = 4
PRED = 5
ARGS = 6

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''             TRANSFORM SDP FILE TO  DAG FORMAT              '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class FileDecoder:

    def __init__(self, file_name, action):
        self.file_name = file_name
        '''Can be train | test | dev'''
        self.action = action
        self.sentences = []
        print("Reading sdp file...")
        self.decode_file()
        print("Done reading sdp file!")

    def decode_file(self):
        f = open(self.file_name, encoding="utf8")
        add_sentence_flag = 0
        for row in f:
            if row != "\n":
                '''New graph (add ROOT)'''
                if row[0] == "#":
                    if add_sentence_flag == 1:
                        '''Convert #pred to #id of the head in heads and arcs'''
                        if self.action != "test":
                            for i in range(len(s.heads)):
                                if s.heads[i]:
                                    for j in range(len(s.heads[i])):
                                        if s.heads[i][j] != ROOT_ID:
                                            s.heads[i][j] = l_pred[s.heads[i][j] - 1]

                            for k in range(len(s.gold_sdp_arcs)):
                                if s.gold_sdp_arcs[k][ID] != ROOT_ID:
                                    s.gold_sdp_arcs[k] = (l_pred[s.gold_sdp_arcs[k][ID] - 1], s.gold_sdp_arcs[k][1],
                                                          s.gold_sdp_arcs[k][2])
                        self.sentences.append(s)
                    add_sentence_flag = 1
                    s = Sentence()
                    '''List with the number of predicates'''
                    l_pred = []
                    s.form.append('ROOT')
                    s.pos.append('ROOT_POS')
                    s.heads.append([])

                else:
                    splited = row.rstrip().split("\t")
                    s.form.append(splited[FORM])
                    s.pos.append(splited[POS])
                    s.heads.append([])

                    '''Add arcs and heads information'''
                    if self.action != "test":
                        '''If the token is the srtructural root, make it dependant of ROOT'''
                        if splited[TOP] == "+":
                            s.gold_sdp_arcs.append((ROOT_ID, int(splited[ID]), "ROOT"))
                            s.heads[-1].append(ROOT_ID)
                        if splited[PRED] == "+":
                            '''Add the id to the list of predicates if it is a predicate'''
                            l_pred.append(int(splited[ID]))
                        '''Pred number which is not the same of the ID'''
                        pred_id = 1
                        for arg in splited[ARGS:]:
                            if arg != "_":
                                s.heads[-1].append(pred_id)
                                s.gold_sdp_arcs.append((pred_id, int(splited[ID]), arg))
                            pred_id += 1

        '''Needed to add last sentence'''
        if add_sentence_flag == 1:
            '''Convert #pred to #id of the head in heads and arcs'''
            if self.action != "test":
                for i in range(len(s.heads)):
                    if s.heads[i]:
                        for j in range(len(s.heads[i])):
                            if s.heads[i][j] != ROOT_ID:
                                s.heads[i][j] = l_pred[s.heads[i][j] - 1]

                for k in range(len(s.gold_sdp_arcs)):
                    if s.gold_sdp_arcs[k][ID] != ROOT_ID:
                        s.gold_sdp_arcs[k] = (l_pred[s.gold_sdp_arcs[k][ID] - 1], s.gold_sdp_arcs[k][1],
                                              s.gold_sdp_arcs[k][2])
            self.sentences.append(s)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''         CONTAINS THE DAG STRUCTURE OF THE SENTENCE         '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class Sentence:

    def __init__(self):
        """Lists with the correspondent field in the index of the token
        [0]: ROOT ; [1]: First word ... [-1] Last word             """
        self.form = []
        self.pos = []
        self.heads = []
        self.gold_sdp_arcs = []