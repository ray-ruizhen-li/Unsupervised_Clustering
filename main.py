# Created by RayLi

from src.data.make_dataset import load_and_preprocess_data
from src.models.train_models import cluster
from src.visulization.visulize import plot_scatter, elbow, silhouette

#from src.models.predict_model import evaluate_model

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "src/data/raw/mall_customers.csv"
    df = load_and_preprocess_data(data_path)

    cldf = cluster(df, 'Annual_Income','Spending_Score', number_cluster=5)
    plot_scatter(df)
    
    ebdf = elbow(df,3,9,'Annual_Income','Spending_Score')
    
    sldf = silhouette(df,3,9,'Annual_Income','Spending_Score')
