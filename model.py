import pandas as pd
import pickle
import statsmodels.api as sm
import argparse


def prep_data(df):
    df = df.rename(columns={'SibSp': 'family_members', 'Parch': 'parents', 'Embarked': 'port'})
    df = df[['Pclass', 'Sex', 'Age', 'family_members', 'parents', 'Fare', 'port']]
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    # Create dummies manually
    df['Pclass_2'] = (df['Pclass'] == 2).astype(int)
    df['Pclass_3'] = (df['Pclass'] == 3).astype(int)
    df['Sex_male'] = (df['Sex'] == 'male').astype(int)
    df['port_Q'] = (df['port'] == 'Q').astype(int)
    df['port_S'] = (df['port'] == 'S').astype(int)

    df = df.drop(['Pclass', 'Sex', 'port'], axis=1)
    df = sm.add_constant(df)

    df = df[['Age', 'family_members', 'parents', 'Fare', 'Pclass_2', 'Pclass_3', 'Sex_male', 'port_Q', 'port_S']]
    return df
    

def load_model(model_name):
    with open(model_name, 'rb') as file:
        model = pickle.load(file)
    return model



## Use below code, if using model.py as the main function.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Titanic Survival Model')
    parser.add_argument('--data', type=str, default='titanic.csv', required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    data = pd.read_csv(args.data)
    model = load_model(args.model)
    score = prep_data(data)
    preds = model.predict(score)
    
    preds = ['S' if pred == 1 else 'NS' for pred in preds ]
    
    pd.DataFrame({'survive_p':preds}).to_csv('predictions.csv', index=False)
    print(preds)
    
# python score.py --data data/titanic.csv --model models/logreg_model.pkl