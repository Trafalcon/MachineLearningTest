using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MachineLearningTest
{
    //Main Class
    class MachineLearningMain
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            MLContext mLContext = new MLContext(seed: 0);
            var model = Train(MLContext, _trainDataPath);
        }

        public static ITransformer Train(MLContext mLContext, string dataPath)
        {
            IDataView dataView = mLContext.Data.LoadFromTextFile<SalaryData>(dataPath, hasHeader: true, separatorChar: ',');
        }
    }

     class SalaryData
    {
        [LoadColumn(0)]
        public float YearsExperience;

        [LoadColumn(1)]
        public float Salary;
    }

     class SalaryDataPrediction
    {
        [ColumnName("Score")]
        public float Salary;
    }
        
}
