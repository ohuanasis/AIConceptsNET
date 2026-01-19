using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace SentimentAnalysis.Classification
{
    internal class TextData
    {
        [LoadColumn(0)]
        public string text { get; set; }
    }
}
