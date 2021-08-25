# datasets import load_dataset
import pandas as pd
import os
class DataLoader:
    def load_sts_dataset(self,filename):
        # Loads a subset of the STS dataset into a DataFrame. In particular both
        # sentences and their human rated similarity score.
        sent_pairs = []
        with open(filename,encoding='utf-8',mode= "r") as f:
            for line in f:
                ts = line.strip().split("\t")
                sent_pairs.append((ts[5], ts[6], float(ts[4])))
        return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])

    def load_sts(self):
        sts_dev = self.load_sts_dataset(os.path.join("data/stsbenchmark", "sts-dev.csv"))
        sts_test = self.load_sts_dataset(os.path.join("data/stsbenchmark", "sts-test.csv"))
        sts_train = self.load_sts_dataset(os.path.join("data/stsbenchmark", "sts-train.csv"))

        return sts_train,sts_dev,sts_test
    def load_sick_datatset(self,file):
        with open(file) as f:
            lines = f.readlines()[1:]
            lines = [l.split("\t") for l in lines if len(l) > 0]
            lines = [l for l in lines if len(l) == 5]

            df = pd.DataFrame(lines, columns=["idx", "sent_1", "sent_2", "sim", "label"])
            df['sim'] = pd.to_numeric(df['sim'])
            return df
    def load_sick(self,):
        sick_train =  self.load_sick_datatset(os.path.join("e:data", "SICK_train.txt"))
        sick_dev  =  self.load_sick_datatset(os.path.join("e:data", "SICK_trial.txt"))
        sick_test  =  self.load_sick_datatset(os.path.join("e:data", "SICK_test_annotated.txt"))
        return sick_train,sick_dev,sick_test
