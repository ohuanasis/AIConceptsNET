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

            /*
            string[] featureColumns = { "SquareFeet", "Bedrooms" };
            string labelColumn = "Price";

            var pipeline = mlContext.Transforms.Concatenate("Features", featureColumns)
                .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: labelColumn));

            var model = pipeline.Fit(data);
            var prediction = model.Transform(data);
            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: labelColumn);
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
            */

            var dataPipeline = mlContext.Transforms.Conversion.ConvertType("SquareFeet", outputKind: DataKind.Single)
                .Append(mlContext.Transforms.NormalizeMinMax("SquareFeet"))
                .Append(mlContext.Transforms.Concatenate("Features", "SquareFeet", "Bedrooms"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood"));

            var transformedData = dataPipeline.Fit(data).Transform(data);
            var transformedDataEnumerable = mlContext.Data.CreateEnumerable<TansformedHousingData>(transformedData, reuseRowObject: false).ToList();

            foreach (var item in transformedDataEnumerable)
            {
                Console.WriteLine($"SquareFeet: {item.SquareFeet}, Bedrooms: {item.Bedrooms}, Price: {item.Price}, Features: {string.Join(", ", item.Features)}, Neighborhood: {string.Join(", ", item.Neighborhood)}");
            }

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
        [LoadColumn(3)]
        public string Neighborhood { get; set; }

    }

    internal class HousingPrediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }

    internal class TansformedHousingData
    {
        public float SquareFeet { get; set; }
        public float Bedrooms { get; set; }
        public float Price { get; set; }
        public float[] Features { get; set; }
        public float[] Neighborhood { get; set; }

    }
}
