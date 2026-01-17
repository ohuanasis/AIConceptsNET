using Microsoft.ML;
using Microsoft.ML.Data;

namespace MachineLearningExample
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //-------------------------------using housing-data.csv file------------------------------------

            //MLContext mlContext = new MLContext();

            //IDataView data = mlContext.Data.LoadFromTextFile<HousingData>("housing-data.csv", separatorChar:',', hasHeader:true);

            //version 1
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

            //version 2
            /*
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
            */

            //-------------------------------using data.csv file------------------------------------

            var context = new MLContext();
            IDataView data = context.Data.LoadFromTextFile<DataPoint>("data.csv", separatorChar:',', hasHeader:true);

            var trainTestSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);

            var logisticRegressionPipeline = context.Transforms.Concatenate("Features", "Feature1", "Feature2")
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", maximumNumberOfIterations: 100));

            var fastTreePipeline = context.Transforms.Concatenate("Features", "Feature1", "Feature2")
                .Append(context.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", numberOfLeaves: 50, numberOfTrees: 100));

            Console.WriteLine("Training logistic Regression model...");
            var logisticRegressionModel = logisticRegressionPipeline.Fit(trainTestSplit.TrainSet);

            Console.WriteLine("Training FastTree model...");
            var fastTreeModel = fastTreePipeline.Fit(trainTestSplit.TrainSet);

            Console.WriteLine("Evaluating logistic Regression model...");
            var logisticRegressionPredictions = logisticRegressionModel.Transform(trainTestSplit.TestSet);
            var logisticRegressionMetrics = context.BinaryClassification.Evaluate(logisticRegressionPredictions);
            EvaluateMetrics("Logistic Regression", logisticRegressionMetrics);

            Console.WriteLine("Evaluating FastTree model...");
            var fastTreePredictions = fastTreeModel.Transform(trainTestSplit.TestSet);
            var fastTreeMetrics = context.BinaryClassification.Evaluate(fastTreePredictions);
            EvaluateMetrics("FastTree", fastTreeMetrics);

            if (logisticRegressionMetrics.Accuracy > fastTreeMetrics.Accuracy)
            {
                Console.WriteLine("Logistic Regression is the best model");
            }
            else if(logisticRegressionMetrics.Accuracy < fastTreeMetrics.Accuracy)
            {
                Console.WriteLine("FastTree is the best model");
            }
            else
            {
                Console.WriteLine("Logistic Regression and FastTree are equally as good");
            }

        }

        private static void EvaluateMetrics(string modelName, BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"{modelName} - Accuracy: {metrics.Accuracy:0.0##}");
            Console.WriteLine($"{modelName} - AUC: {metrics.AreaUnderRocCurve:0.0##}");
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

    internal class DataPoint
    {
        [LoadColumn(0)]
        public float Feature1 { get; set; }
        [LoadColumn(1)]
        public float Feature2 { get; set; }
        [LoadColumn(2)]
        public bool Label { get; set; }
    }

    internal class Prediction
    {
        [ColumnName("Score")]
        public float Score { get; set; }
        [ColumnName("Probability")]
        public float Probability { get; set; }
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
