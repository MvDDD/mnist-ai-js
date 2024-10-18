let gpu = new GPU.GPU()

let net = new NeuralNet.NeuralNet(
    [784,19,10],
    (a)=>Math.max(a, a/10),
    (a)=>Math.tanh(a)
    )
    
let gpuTrainer = new NeuralNet.GPUNet(net,50,"per_data",gpu)
gpuTrainer.setInput([1, 2, 3])
gpuTrainer.run(2)