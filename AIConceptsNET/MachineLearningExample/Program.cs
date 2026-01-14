using Microsoft.ML;
using Microsoft.ML.Data;

namespace MachineLearningExample
{
    internal class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            IDataView data = mlContext.Data.LoadFromTextFile<HousingData>("housing-data.csv", separatorChar:',', hasHeader:true);

            string[] featureColumns = { "SquareFeet", "Bedrooms" };
            string labelColumn = "Price";

            var pipeline = mlContext.Transforms.Concatenate("Features", featureColumns)
                .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: labelColumn));

            var model = pipeline.Fit(data);
            var prediction = model.Transform(data);
            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: labelColumn);
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
        }
    }

    internal class HousingData
    {
        [LoadColumn(0)]
        public float SquareFeet { get; set; }
        [LoadColumn(1)]
        public float Bedrooms { get; set; }
        [LoadColumn(2)]
        public float Price { get; set; }

    }

    internal class HousingPrediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }
}
