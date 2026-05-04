import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDAVisualizer:
    def __init__(self, csv_path="src/data/processed/merged_data.csv", out_dir="reports/figures"):

        # set up input/output folder path
        self.csv_path = csv_path
        self.out_dir = out_dir
        self.df = None
        self.numerical_cols = None
        self.categorical_cols = None

        sns.set_theme(style="whitegrid")
        os.makedirs(self.out_dir, exist_ok=True)

    def load(self):

        self.df = pd.read_csv(self.csv_path)

        #columns to exclude, not features that need to be visualized
        skip = {"ID", "default", "status_0", "status_1", "status_2",
                "status_3", "status_4", "status_5", "status_C", "status_X",
                "FLAG_MOBIL"}

        self.numerical_cols = []
        for i in self.df.select_dtypes(include="number").columns:
            if i not in skip:
                self.numerical_cols.append(i)

        self.categorical_cols = []
        for j in self.df.select_dtypes(include="object").columns:
            if j not in skip:
                self.categorical_cols.append(j)

        print(f"Loaded data: {self.df.shape}")
        return self.df

    def save(self, fname):

        path = os.path.join(self.out_dir, fname)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved: {path}")

    def summary_stats(self):

        #show stats of numerical columns (count, mean, std, min, max)
        stats = self.df[self.numerical_cols].describe().T.round(2)

        fig, ax = plt.subplots(figsize=(14, max(4, len(stats) * 0.5)))
    
        ax.axis("off")

        t = ax.table(
            cellText=stats.values,
            colLabels=stats.columns,
            rowLabels=stats.index,
            cellLoc="center",
            loc="center"
        )

        t.set_fontsize(8)
        t.scale(1.2, 1.5)
        ax.set_title("Statistics of all Numerical Features", fontsize=13, fontweight="bold", pad=20)
        
        self.save("summary_statistics.png")

    def histograms(self):

        #histogram for each numerical feature to find for skewed features and outliers
        for col in self.numerical_cols:
            fig, ax = plt.subplots(figsize=(6, 4))

            ax.hist(self.df[col].dropna(), bins=30, color="steelblue", edgecolor="white")
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            self.save(f"hist_{col}.png")

    def bar_charts(self):

        #plots bar chart for each categorical feature
        for col in self.categorical_cols:
            counts = self.df[col].value_counts()

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(counts.index, counts.values, color="steelblue", edgecolor="white")
            ax.set_title(col)
            ax.set_xlabel(col)

            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=30)
            self.save(f"bar_{col}.png")

    def correlation_heatmap(self):

        #correlation matrix for see which numerical features are most correlated with each other 
        cols = self.numerical_cols + ["default"]

        corr = self.df[cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        self.save("correlation_heatmap.png")

    def target_distribution(self):

        #how many customers defaulted and didnt default 
        counts = self.df["default"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(["No Default", "Default"], counts.values, color=["steelblue", "tomato"])
        ax.set_title("Target Distribution")
        ax.set_ylabel("Count")
        self.save("target_distribution.png")

    def boxplots(self):

        #boxplot for each numerical feature by default/non default status
        for col in self.numerical_cols:
            fig, ax = plt.subplots(figsize=(6, 4))

            self.df.boxplot(column=col, by="default", ax=ax)
            ax.set_title(col)

            ax.set_xlabel("Default (0=No, 1=Yes)")
            ax.set_ylabel(col)
            plt.suptitle("")
            self.save(f"box_{col}.png")

    def run(self):
        
        self.load()
        self.summary_stats()
        self.histograms()
        self.bar_charts()
        self.correlation_heatmap()
        self.target_distribution()
        self.boxplots()

        print("Figures saved to:", self.out_dir)


if __name__ == "__main__":
    viz = EDAVisualizer()
    viz.run()