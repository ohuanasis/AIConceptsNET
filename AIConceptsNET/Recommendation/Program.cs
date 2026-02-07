using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Recommendation
{
    internal class Program
    {
        static void Main(string[] args)
        {
            #region Creating data for training the model
            MLContext mlContext = new MLContext();
            IDataView fullData = mlContext.Data.LoadFromTextFile<MovieRating>("ratings.csv", hasHeader: true, separatorChar: ',');
            IDataView preprocessedData = PreprocessData(mlContext, fullData);
            SaveData(mlContext, preprocessedData, "preprocessed_ratings.csv");
            (IDataView trainingDataView, IDataView testDataView) data = LoadData(mlContext);
            PrintDataPreview(data.trainingDataView);
            PrintDataPreview(data.testDataView);
            #endregion

            #region training the model
            ITransformer model = TrainModel(mlContext, data.trainingDataView);
            #endregion

            #region evaluating the model and test the model
            EvaluateModel(mlContext, data.testDataView, model);
            UseModelForSinglePrediction(mlContext, model);
            #endregion
        }

        public static IDataView PreprocessData(MLContext mlContext, IDataView dataView)
        {
            return mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userId", inputColumnName: "userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieId",
                    inputColumnName: "movieId"))
                .Fit(dataView)
                .Transform(dataView);
        }

        public static void SaveData(MLContext mlContext, IDataView dataView, string dataPath)
        {
            using (var fileStream = new FileStream(dataPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Data.SaveAsText(dataView, fileStream, separatorChar: ',', headerRow: true, schema: false);
            }
        }

        public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            var dataPath = Path.Combine(Environment.CurrentDirectory, "preprocessed_ratings.csv");
            IDataView fullData =
                mlContext.Data.LoadFromTextFile<MovieRating>(dataPath, hasHeader: true, separatorChar: ',');
            var trainTestData = mlContext.Data.TrainTestSplit(fullData, testFraction: 0.2);
            IDataView trainData = trainTestData.TrainSet;
            IDataView testData = trainTestData.TestSet;
            return (trainData, testData);
        }

        public static void PrintDataPreview(IDataView dataView)
        {
            var preview = dataView.Preview();
            foreach (var row in preview.RowView)
            {
                foreach (var column in row.Values)
                {
                    Console.Write($"{column.Key}: {column.Value} ");
                }

                Console.WriteLine();
            }
        }

        public static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: "outputUserId", inputColumnName: "userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "outputMovieId",
                    inputColumnName: "movieId"));

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "outputUserId",
                MatrixRowIndexColumnName = "outputMovieId",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            ITransformer model = trainerEstimator.Fit(trainingDataView);

            Console.WriteLine("Model successfully trained.");

            return model;
        }

        public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            var predictions = model.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
            Console.WriteLine($"R-squared: {metrics.RSquared}");
        }

        public static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
            var testInput = new MovieRating { userId = 14, movieId = 433 };
            var movieRatingPrediction = predictionEngine.Predict(testInput);
            Console.WriteLine($"Predicted rating for user {testInput.userId} and movie {testInput.movieId}: {Math.Round(movieRatingPrediction.Score, 1)}");

            string recommendation = Math.Round(movieRatingPrediction.Score, 1) >= 3.5 
                ? $"Movie {testInput.movieId} is recommended for user {testInput.userId}" : $"Movie {testInput.movieId} is not recommended for user {testInput.userId}";

            Console.WriteLine($"Recommendation: {recommendation}");
        }
    }

    public class MovieRating
    {
        [LoadColumn(0)]
        public float userId;
        [LoadColumn(1)]
        public float movieId;
        [LoadColumn(2)]
        public float Label;

    }

    public class MovieRatingPrediction
    {
        public float Label;
        public float Score;
    }
}