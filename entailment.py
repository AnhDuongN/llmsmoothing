from common import dataset
import pandas as pd

if __name__ == "__main__":
    """
    Exact matching to see if the ball centers correspond to the original answer(s) 
    from the dataset.
    """
    list_answers = []
    df = pd.read_csv("output.csv")
    df['normalized_answer'] = dataset['normalized_answer']
    df['entail'] = df.apply(lambda x: x['answer'] in x['normalized_answer'], axis=1)
    df.to_csv('entailment.csv')  
    