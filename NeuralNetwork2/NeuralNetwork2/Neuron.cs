using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork2
{
    public class Neuron
    {
        public List<Synapse> Inputs { get; set; }
        public List<Synapse> Outputs { get; set; }
        public double Output { get; set; }
        public double TargetOutput { get; set; }
        public double Delta { get; set; }
        public double Bias { get; set; }
        int? maxInput { get; set; }
        public NeuronType NeuronType { get; set; }

        public bool AcceptConnection
        {
            get
            {
                return !(NeuronType == NeuronType.Hidden && maxInput.HasValue && Inputs.Count > maxInput);
            }
        }

        public double InputSignal
        {
            get
            {
                return Inputs.Sum(d => d.Weight * (d.Source.Output + Bias));
            }
        }

        public Neuron(NeuronType neuronType, int? maxInput)
        {
            this.NeuronType = neuronType;
            this.maxInput = maxInput;
            this.Inputs = new List<Synapse>();
            this.Outputs = new List<Synapse>();
        }

        public double BackwardSignal()
        {
            if (Outputs.Any())
            {
                Delta = Outputs.Sum(d => d.Target.Delta * d.Weight) * ActivatePrime(Output);
            }
            else
            {
                Delta = (Output - TargetOutput) * ActivatePrime(Output);
            }

            return Delta + Bias;
        }

        public void AdjustWeights(double learnRate, double momentum)
        {
            if(Inputs.Any())
            {
                foreach(var synp in Inputs)
                {
                    var adjustDelta = Delta * synp.Source.Output;
                    synp.Weight -= learnRate * adjustDelta + synp.PreDelta * momentum;
                    synp.PreDelta = adjustDelta;
                }
            }
        }

        public double ForwardSignal()
        {
            Output = Activate(InputSignal);

            return Output;
        }

        private double ActivatePrime(double x)
        {
            return x * (1 - x);
        }

        private double Activate(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }
    }
}
