ROOT_ID = 0


class FileEncoder:

    def __init__(self, sentences, file_input, file_output):
        self.sentences = sentences
        self.file_input = file_input
        self.file_output = file_output
        self.decode_file()

    def decode_file(self):
        input = open(self.file_input, "r")
        output = open(self.file_output, "w")

        s = -1
        for row in input:
            if row != "\n":
                if row[0] == "#":
                    output.write(row)
                    token_id = 0
                    s += 1
                    top = set()
                    preds = set()

                    '''Search for structural roots and for preds'''
                    for arc in self.sentences[s].arcs:
                        if arc[0] == ROOT_ID:
                            top.add(arc[1])
                        else:
                            preds.add(arc[0])

                    preds = list(preds)
                    preds.sort()

                else:
                    token_id += 1
                    args = []
                    for arc in self.sentences[s].arcs:
                        if token_id == arc[1] and arc[0] != 0:
                            args.append((preds.index(arc[0]), arc[2]))
                    output.write(row.rstrip("\n"))

                    '''Write top and pred'''
                    if token_id in top:
                        output.write("\t+")
                    else:
                        output.write("\t-")
                    if token_id in preds:
                        output.write("\t+")
                    else:
                        output.write("\t-")

                    for p in range(len(preds)):
                        p_flag = 0
                        for a in args:
                            if p == a[0]:
                                output.write("\t" + a[1])
                                p_flag = 1
                                break
                        if p_flag == 0:
                            output.write("\t_")
                    output.write("\n")
            else:
                output.write("\n")

        input.close()
        output.close()






