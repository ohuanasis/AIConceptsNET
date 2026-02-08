using Microsoft.ML;
using Microsoft.ML.Data;

namespace SentimentAnalysisExample
{
    internal class Program
    {
        static void Main(string[] args)
        {
            #region SetUp data for sentiment analysis AI
            var mlContext = new MLContext();
            string dataPath = Path.Combine(Environment.CurrentDirectory, "movieReviews.csv");
            string text = File.ReadAllText(dataPath);
            using (StreamReader sr = new StreamReader(dataPath))
            {
                text = text.Replace("\'", "");
                text = text.Replace("positive", "true");
                text = text.Replace("negative", "false");
            }
            File.WriteAllText(dataPath, text);
            #endregion

            #region load the pre-processing data in preperation for training the model
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(dataPath, hasHeader: true, allowQuoting: true, separatorChar: ',');
            Console.WriteLine("Data loaded successfully:");
            Console.WriteLine();
            var preview = dataView.Preview(maxRows: 5);
            foreach (var row in preview.RowView)
            {
                foreach (var column in row.Values)
                {
                    Console.WriteLine($"{column.Key}: {column.Value}");
                }
            }
            #endregion

            #region Train the model
            var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainData = trainTestSplit.TrainSet;

            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "review")
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));
            var model = pipeline.Fit(trainData);
            #endregion

            #region Evaluate and test the model
            var testData = trainTestSplit.TestSet;
            var predictions = model.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
            Console.WriteLine($"Log Loss: {metrics.LogLoss:F2}");  
            #endregion

        }
    }

    public class SentimentData {
        [LoadColumn(0)]
        public string review { get; set; }

        [LoadColumn(1)]
        [ColumnName("Label")]
        public bool sentiment { get; set; }
    }
}
