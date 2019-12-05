from ivpy import discretize
import seaborn as sns

if __name__ == "__main__":

    d = sns.load_dataset('titanic')

    discretize(d['fare'], d['survived'], maxbin=6)