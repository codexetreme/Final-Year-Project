from pythonrouge.pythonrouge import Pythonrouge
# import rouge
# system summary(predict) & reference summary
#summary = [[" Tokyo is the one of the biggest city in the world."]]
#reference = [[["The capital of Japan, Tokyo, is the center of Japanese economy."]]]

#summary=
#reference=highlights
class Reward():
    def __init__(self):
        # self.pythonrouge()
        print ('yoyo')
    # def pythonrouge(self):
    #     # initialize setting of ROUGE to eval ROUGE-1, 2, SU4
    #     # if you evaluate ROUGE by sentence list as above, set summary_file_exist=False
    #     # if recall_only=True, you can get recall scores of ROUGE
    #     # TODO: Match these parameters to their Rouge parameters
    #     self.rouge = Pythonrouge(summary_file_exist=False,
    #                     summary=summary, reference=reference,
    #                     n_gram=3, ROUGE_SU4=False, ROUGE_L=True,
    #                     recall_only=True, stemming=True, stopwords=True,
    #                     word_level=True, length_limit=True, length=50,
    #                     use_cf=False, cf=95, scoring_formula='average',
    #                     resampling=True, samples=1000, favor=True, p=0.5)

    def get_reward(self,predicted_summ,gold_summ):
        summary=gold_summ
        reference=predicted_summ
        self.rouge = Pythonrouge(summary_file_exist=False,
                        summary=summary, reference=reference,
                        n_gram=3, ROUGE_SU4=False, ROUGE_L=True,
                        recall_only=True, stemming=True, stopwords=True,
                        word_level=True, length_limit=True, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
        self.rouge.summary = gold_summ
        self.rouge.reference = predicted_summ
        score = self.rouge.calc_score()
        score=score['ROUGE-1'] + score['ROUGE-2']*5 + score['ROUGE-3']*2 +score['ROUGE-L']*2
        return score
    


# print (get_reward( [['hello shanky, are you going to japan' ]],[[['The capital of Japan, Tokyo, is the hello of Japanese economy.']]]))
