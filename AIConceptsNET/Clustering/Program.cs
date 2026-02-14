using Microsoft.ML;
using Microsoft.ML.Data;

namespace Clustering
{
    internal class Program
    {
        private static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "customers.csv");
        static void Main(string[] args)
        {
            #region Load the data
            MLContext mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromTextFile<CustomerData>(dataPath, separatorChar: ',', hasHeader: true);
            #endregion

            #region Train Model
            var pipeline = mlContext.Transforms.Concatenate("Features", "AnnualIncome", "SpendingScore")
                .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));

            var model = pipeline.Fit(dataView);
            var transformedData = model.Transform(dataView);
            #endregion

            #region Evaluate and test the model
            var predictions = mlContext.Data.CreateEnumerable<ClusterPrediction>(transformedData, reuseRowObject: false);
            foreach (var prediction in predictions)
            {
                Console.WriteLine($"CustomerID: {prediction.CustomerID}, Predicted Cluster: {prediction.PredictedClusterId}");
            } 
            #endregion
        }
    }

    public class CustomerData {
        [LoadColumn(0)]
        public float CustomerID;

        [LoadColumn(1)]
        public float AnnualIncome;

        [LoadColumn(2)]
        public float SpendingScore;
    }

    public class ClusterPrediction {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId { get; set; }

        [ColumnName("Score")]
        public float[] Score { get; set; }

        public float CustomerID { get; set; }
    }
}
