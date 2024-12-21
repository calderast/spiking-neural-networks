# spiking-neural-networks

Code associated with *Spiking Neural Networks: Considerations, Implementations, and Comparison with Convolutional Neural Networks*. Jackson Borchardt, Samantha Coury, Stephanie Crater, Ashley Qin. VS 265, Fall 2024

See [Neuron_Model_Comparison.ipynb](Neuron_Model_Comparison.ipynb) for a comparison of a variety of neuron models: standard **Leaky Integrate-and-Fire (LIF)** neuron, **Hodgkin-Huxley** neuron, various **Izhikevich** neurons (regular spiking, intrinsically bursting, chattering, or fast spiking), and our novel **Funky** neuron (equivalent to an LIF neuron with threshold that randomly changes when a spike is fired).

See [LIF_vs_Funky_snn.ipynb](LIF_vs_Funky_snn.ipynb) for a comparison of spiking neural networks comprised of snntorch's Leaky (LIF) neuron vs. our custom Funky neuron. Note that this notebook requires a custom version of snnTorch with our `Funky` neuron added, available at [https://github.com/calderast/snntorch]. The pull request detailing the exact functionality added to snnTorch is [here](https://github.com/calderast/snntorch/pull/1).

See [cnn_mnist.py](cnn_mnist.py) for a implementation of a simple feedforward convolutional neural network (CNN) trained such that it achieved roughly equivalent (~94%) test set accuracy to our SNNs. 

Much of our code is adapted from [snntorch](https://github.com/jeshraghian/snntorch). Relevant snntorch tutorials are included in the snntorch_tutorials/ folder.