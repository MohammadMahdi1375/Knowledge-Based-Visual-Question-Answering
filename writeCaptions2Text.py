class  writeCaptions2Text:
    def __init__(self, file_adr='./captions.text'):
        self.file_adr = file_adr
        with open(self.file_adr, 'w') as file:
            pass

        with open("./Results.text", 'w') as file2:
            file2.write("%-10s %-10s %-10s\n" % ("n_iter", "n_true1", "n_true2"))
            file2.write("---------- ---------- ---------\n")


    def write(self, Q_A_Cap_QA, captions, Question=None, Answer=None, Candidate_answer=None, captions_Aux_Information=None):
        with open(self.file_adr, 'a') as file:
            """if Question != None:
                file.write(Question + '\n')"""

            if Candidate_answer != None:
                file.write("Ground Truth ----> " + str(Candidate_answer) + "\n")

            if Answer != None:
                file.write("Generated Answer ----> " + str(Answer) + '\n')
            
            if captions_Aux_Information != None:
                file.write("Question Concept ----> " + captions_Aux_Information + '\n')

            """for caption in captions:
                file.write(caption + '\n')"""
            file.write(Q_A_Cap_QA + '\n')
            file.write('='*150 + '\n')
    

    def writeResults(self, n_iter, n_true1, n_true2):
        with open("./Results.text", 'a') as file2:
            file2.write("%-10d %-10d %-10d\n" % (n_iter, n_true1, n_true2))

