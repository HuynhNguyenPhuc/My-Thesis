import pandas as pd
from launch_callable import run_training_with_prompt
import ast

def main():
    test_file = "test.csv"
    test_df = pd.read_csv(test_file)
    test_df["captions"] = test_df["captions"].apply(ast.literal_eval)
    test_df["captions"] = test_df["captions"].apply(lambda x: x[0])
    test_df = test_df.iloc[:100]

    for idx, row in test_df.iterrows():
        prompt = row["captions"]
        prompt = prompt.replace(":", " with")

        print(f"Running training for row {idx} with prompt: {prompt}")

        run_training_with_prompt(prompt)

if __name__ == "__main__":
    main()
