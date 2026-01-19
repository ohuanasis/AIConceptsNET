using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace SentimentAnalysis.Classification
{
    internal class MovieReview
    {
        [LoadColumn(0)]
        public string text { get; set; }
        [LoadColumn(1), ColumnName("Label")]
        public bool sentiment { get; set; }
    }
}
