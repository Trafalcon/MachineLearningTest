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

     class TaxiTrip
    {
        [LoadColumn(0)]
        public string VendorId;

        [LoadColumn(1)]
        public string RateCode;

        [LoadColumn(2)]
        public float PassengerCount;

        [LoadColumn(3)]
        public float TripTime;

        [LoadColumn(4)]
        public float TripDistance;

        [LoadColumn(5)]
        public string PaymentType;

        [LoadColumn(6)]
        public float FareAmount;
    }

     class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
        
}
