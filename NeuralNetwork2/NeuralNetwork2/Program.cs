using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork2
{
    class Program
    {
        private static int inputOutputScalingFactor = 100;

        static void Main(string[] args)
        {
            NeuralNetwork network = new NeuralNetwork(2, new int[] { 3, 3 }, 1, null);
            List<double[]> trainingData = new List<double[]>();
            int trainingDataSize = 150;

            for(int i = 0; i < trainingDataSize; i++)
            {
                double input1 = NeuralNetwork.GetCryptographicallyRandomDouble();
                double input2 = NeuralNetwork.GetCryptographicallyRandomDouble();

                double output = Program.DesiredFunction(input1, input2);

                trainingData.Add(new double[] { input1, input2, output });
            }

            Console.WriteLine("Neural Network is training");
            Program.Train(100000, trainingData, network);
            Console.WriteLine();
            Console.WriteLine("Training complete. Press any key to perform calculation.");
            Console.ReadLine();

            var inputData = new double[] { 0.3, 0.2 };
            var result = network.FeedForward(inputData);

            Console.Write($"Inputs {inputData[0]} {inputData[1]}    Output: {result[0] * inputOutputScalingFactor}");
            Console.ReadLine();
        }

        public static void Train(int times, List<double[]> trainingData, NeuralNetwork network)
        {
            for (int i = 0; i < times; i++)
            {
                if(i % 100 == 0)
                    Console.Write(".");

                var shuffledTrainingData = trainingData.OrderBy(d => NeuralNetwork.GetCryptographicallyRandomDouble()).ToList();
                List<double> errors = new List<double>();

                foreach (var item in shuffledTrainingData)
                {
                    double[] inputs = new double[] { item[0], item[1] };
                    double[] output = new double[] { item[2] };

                    network.Train(inputs, output);

                    errors.Add(network.GlobalError);
                }
            }
        }

         public static double DesiredFunction(double input1, double input2)
        {
            double result;

            result = Math.Abs(input2 + input1) / inputOutputScalingFactor;

            return result;
        }
    }
}
