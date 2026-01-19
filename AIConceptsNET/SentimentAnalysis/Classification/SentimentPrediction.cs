using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace SentimentAnalysis.Classification
{
    internal class SentimentPrediction
    {
        [ColumnName("Score")]
        public float SentimentScore { get; set; }
        public bool IsPositiveSentiment => SentimentScore < 0.5f;
    }
}
