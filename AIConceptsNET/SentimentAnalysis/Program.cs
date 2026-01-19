using Microsoft.ML;
using SentimentAnalysis.Classification;

namespace SentimentAnalysis
{
    internal class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            string dataPath = "movieReviewsTraining.csv";
            string text = File.ReadAllText(dataPath);

            using (StreamReader sr = new StreamReader(dataPath)) 
            {
                text = text.Replace("\'", "");
            }

            File.WriteAllText(dataPath, text);

            IDataView dataView = mlContext.Data.LoadFromTextFile<MovieReview>(dataPath, hasHeader: true, allowQuoting: true, separatorChar: ',');

            var preview = dataView.Preview();

            foreach (var row in preview.RowView)
            {
                Console.WriteLine($"{row.Values[0]} | {row.Values[1]}");
            }

        }
    }

}