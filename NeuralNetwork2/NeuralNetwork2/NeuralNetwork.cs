using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork2
{
    public class NeuralNetwork
    {
        public double LearnRate = .5;
        public double Momentum = .3;
        public List<Layer> Layers { get; private set; }
        int? maxNeuronConnection;
        //public int? Seed { get; set; }
        public NeuralNetwork(int inputs, int[] hiddenLayers, int outputs, int? maxNeuronConnection = null)
        {
            //this.Seed = seed;
            this.maxNeuronConnection = maxNeuronConnection;
            this.Layers = new List<Layer>();
            buildLayer(inputs, NeuronType.Input);
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                buildLayer(hiddenLayers[i], NeuronType.Hidden);
            }
            buildLayer(outputs, NeuronType.Output);
            InitSynapses();

        }

        void buildLayer(int nodeSize, NeuronType neuronType)
        {
            var layer = new Layer();
            var nodeBuilder = new List<Neuron>();
            for (int i = 0; i < nodeSize; i++)
            {
                nodeBuilder.Add(new Neuron(neuronType, maxNeuronConnection));
            }
            layer.Neurons = nodeBuilder.ToArray();
            Layers.Add(layer);
        }

        private void InitSynapses()
        {
            //var rnd = Seed.HasValue ? new Random(Seed.Value) : new Random();

            //var rnd = GetCryptographicallyRandomDouble();

            for (int i = 0; i < Layers.Count - 1; i++)
            {
                var layer = Layers[i];
                var nextLayer = Layers[i + 1];
                foreach (var node in layer.Neurons)
                {
                    node.Bias = 0.1 * GetCryptographicallyRandomDouble();
                    foreach (var nNode in nextLayer.Neurons)
                    {
                        if (!nNode.AcceptConnection) continue;
                        var snypse = new Synapse(GetCryptographicallyRandomDouble(), nNode, node);
                        node.Outputs.Add(snypse);
                        nNode.Inputs.Add(snypse);
                    }
                }
            }

        }

        public static double GetCryptographicallyRandomDouble()
        {
            RNGCryptoServiceProvider rngCsp = new RNGCryptoServiceProvider();
            Byte[] bytes = new Byte[8];
            rngCsp.GetBytes(bytes);

            ulong ul = BitConverter.ToUInt64(bytes, 0) / (1 << 11);
            Double d = ul / (Double)(1UL << 53);

            return d;
        }

        public double GlobalError
        {
            get
            {
                return Math.Round(Layers.Last().Neurons.Sum(d => Math.Pow(d.TargetOutput - d.Output, 2) / 2), 4);
            }
        }

        public void BackPropagation()
        {
            for (int i = Layers.Count - 1; i > 0; i--)
            {
                var layer = Layers[i];
                foreach (var node in layer.Neurons)
                {
                    node.BackwardSignal();
                }
            }
            for (int i = Layers.Count - 1; i >= 1; i--)
            {
                var layer = Layers[i];
                foreach (var node in layer.Neurons)
                {
                    node.AdjustWeights(LearnRate, Momentum);
                }
            }
        }

        public double[] Train(double[] _input, double[] _outputs)
        {
            //if (_outputs.Count() != Layers.Last().Neurons.Count() || _input.Any(d => d < 0 || d > 1) || _outputs.Any(d => d < 0 || d > 1))
                //throw new ArgumentException();

            var outputs = Layers.Last().Neurons;
            for (int i = 0; i < _outputs.Length; i++)
            {
                outputs[i].TargetOutput = _outputs[i];
            }

            var result = FeedForward(_input);

            BackPropagation();
            return result;
        }

        public double[] FeedForward(double[] _input)
        {
            if (_input.Count() != Layers.First().Neurons.Count())
                throw new ArgumentException();


            var InputLayer = Layers.First().Neurons;
            for (int i = 0; i < _input.Length; i++)
            {
                InputLayer[i].Output = _input[i];
            }

            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                foreach (var node in layer.Neurons)
                {
                    node.ForwardSignal();
                }
            }

            return Layers.Last().Neurons.Select(d => d.Output).ToArray();
        }
    }
}
