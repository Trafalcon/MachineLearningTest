using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MachineLearningTest
{
    class MachineLearningMain
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
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
