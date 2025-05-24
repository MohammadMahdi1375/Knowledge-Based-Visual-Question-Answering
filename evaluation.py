

class evaluation:
    def __init__(self):
        pass

    def accuracy(self, predicted_output, true_output):
        total_data = len(true_output)
        n_true = 0

        for i, y_pred in enumerate(predicted_output):
            flag = False
            for y_pre in y_pred:
                if y_pre in true_output[i]:
                    flag = True
                    break
                else:
                    for tr_output in true_output[i]:
                        word1 = set(y_pre.split())
                        word2 = set(tr_output.split())

                        if bool(word1.intersection(word2)):
                            flag = True
                            break
            if (flag):
                n_true += 1



        print(n_true)
        return n_true/total_data
