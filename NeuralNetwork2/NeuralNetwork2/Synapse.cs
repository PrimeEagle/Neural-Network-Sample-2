using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork2
{
    public class Synapse
    {
        public double Weight { get; set; }
        public Neuron Target { get; set; }
        public Neuron Source { get; set; }
        public double PreDelta { get; set; }
        public double Gradient { get; set; }

        public Synapse(double weight, Neuron target, Neuron source)
        {
            Weight = weight;
            Target = target;
            Source = source;
        }
    }
}
