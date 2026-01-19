using Microsoft.ML;
using SentimentAnalysis.Classification;
using System.IO;
using Microsoft.ML.Trainers;

namespace SentimentAnalysis
{
    internal class Program
    {
        static void Main(string[] args)
        {
            #region the following code is for training the model and saving it to a file sentiment_model.zip
            /*
            MLContext mlContext = new MLContext();

            string dataPath = "movieReviewsTraining.csv";
            string text = File.ReadAllText(dataPath);

            using (StreamReader sr = new StreamReader(dataPath)) 
            {
                text = text.Replace("\'", "");
            }

            File.WriteAllText(dataPath, text);

            IDataView dataView = mlContext.Data.LoadFromTextFile<MovieReview>(dataPath, hasHeader: true, allowQuoting: true, separatorChar: ',');

            //Console.WriteLine("Data loaded successfully");
            //Console.WriteLine();

            //var preview = dataView.Preview();
            //foreach (var row in preview.RowView)
            //{
            //    Console.WriteLine($"{row.Values[0]} | {row.Values[1]}");
            //}

            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features","text")
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));

            var model = pipeline.Fit(dataView);
            var predictions = model.Transform(dataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"Precision: {metrics.PositivePrecision}");
            Console.WriteLine($"Recall: {metrics.PositiveRecall}");
            Console.WriteLine($"F1-Score: {metrics.F1Score}");
            mlContext.Model.Save(model, dataView.Schema, "sentiment_model.zip");
            */
            #endregion

            #region the following code is for running the model on test data

            string modelPath = "sentiment_model.zip";
            string testDataPath = "movieReviewsTesting.csv";

            MLContext mlContext = new MLContext();

            ITransformer model;
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                model = mlContext.Model.Load(stream, out var modelInputSchema);
            }

            IDataView testDataView = mlContext.Data.LoadFromTextFile<TextData>(testDataPath, separatorChar: ',', hasHeader:true);
            var predictor = mlContext.Model.CreatePredictionEngine<TextData, SentimentPrediction>(model);

            var testDataList = mlContext.Data.CreateEnumerable<TextData>(testDataView, reuseRowObject: false).ToList();

            foreach (var data in testDataList)
            {
                var prediction = predictor.Predict(data);
                Console.WriteLine($"Text: {data.text} | Positive Sentiment: {prediction.IsPositiveSentiment}");
            }

            #endregion

        }
    }

}