class NeuralNet {
    constructor(layers, act = function(a) { return Math.max(a, 0) }, out = function(a) { return Math.tanh(a) }, random = Math.random) {
        if (layers) {
            if (layers.nodes) {
                this.nodes = layers.nodes;
                this.paths = layers.paths;
                this.act = layers.act;
                this.out = layers.out;
                this.random = random;
                this.err = layers.err;
            } else {
                this.nodes = layers.map(function(i) {
                    return Array(i).fill(0).map(() => 0);
                });
                this.act = act;
                this.out = out;
                this.random = random;
                this.paths = [];
                for (let layer = 0; layer < layers.length - 1; layer++) {
                    let pathLayer = [];
                    for (let start = 0; start < layers[layer]; start++) {
                        for (let end = 0; end < layers[layer + 1]; end++) {
                            pathLayer.push([start, end, 1 / (layers[layer] * layers[layer + 1])]);
                        }
                    }
                    this.paths.push(pathLayer);
                }
                this.err = 0;
            }
        }
        this.outputSize = this.nodes[this.nodes.length - 1].length;
        this.inputSize = this.nodes[0].length;
    }

    mutatePaths(amount) {
        if (!this.limit) {
            this.paths = this.paths.map(function(layer) {
                return layer.map(function(path) {
                    return [path[0], path[1], path[2] + ((Math.random() - 0.5) * amount)];
                });
            });
        } else {
            this.paths = this.paths.map(layer => layer.map(path => {
                let newWeight = path[2] + ((this.random() - 0.5) * amount);
                
                newWeight = Math.max(Math.min(newWeight, this.max), -this.max);
                return [path[0], path[1], newWeight];
            }));
        }
    }

    
    mutateNodes(amount) {
        if (!this.max) {
            this.nodes = this.nodes.map(layer => layer.map(node => node + ((Math.random() - 0.5) * amount)));
        } else {
            this.nodes = this.nodes.map(layer => layer.map(node => {
                let newNode = node + ((this.random() - 0.5) * amount);
                
                newNode = Math.max(Math.min(newNode, this.max), -this.max);
                return newNode;
            }));
        }
    }

    clone() {
        let n = new NeuralNet({ nodes: this.nodes.map(l => l.slice()), paths: this.paths.map(l => l.map(p => p.slice())), act: this.act, out: this.out, err: this.err });
        if (this.max) {
            n.max = this.max;
        }
        return n;
    }

    run(inputs) {
        if (inputs.length !== this.nodes[0].length) {
            throw new TypeError("input size incorrect: " + inputs.size);
        }
        let model = this.clone();
        inputs.forEach((i, j) => model.nodes[0][j] = i);
        model.paths.forEach((layer, layerNum) => {
            layer.forEach(path => {
                model.nodes[layerNum + 1][path[1]] += model.nodes[layerNum][path[0]] * path[2];
            });
            model.nodes[layerNum + 1] = model.nodes[layerNum + 1].map(a => this.act(a));
        });
        return model.nodes.pop().map(a => this.out(a));
    }

    
    toString() {
        return JSON.stringify({
            nodes: this.nodes.map(l => l.map(v => parseFloat(v.toFixed(10)))),
            paths: this.paths.map(l => l.map(p => [p[0], p[1], parseFloat(p[2].toFixed(10))])),
            act: this.act.toString(),
            out: this.out.toString()
        });
    }

    export(){
        return this.toString()
    }

    
    fromString(str) {
        try {
            let m = JSON.parse(str);
            m.act = eval("(() => { return " + m.act + " })()");
            m.out = eval("(() => { return " + m.out + " })()");
            return new NeuralNet(m);
        } catch {
            return this;
        }
    }

    fromFile(path){
        try{
            this.fromString(require("fs").readFileSync())
        } catch {
            return this
        }
    }

    from(net) {
        return net.clone();
    }

    normalize(){
        let normalised = this.clone();

        let maxNode = Math.max(...normalised.nodes.flat());
        let maxPath = Math.max(...normalised.paths.map(l => l.map(p => p[2])).flat());

        normalised.nodes = normalised.nodes.map(
            layer => layer.map(
                node => (node / maxNode)
                )
            );

        normalised.paths.forEach(
            layer => layer.forEach(
                path => path[2] = (path[2] / maxPath) / 4)
            );

        return normalised
    }

    draw(ctx, size = 20) {
        let normalised = this.clone();

        let maxNode = Math.max(...normalised.nodes.flat());
        let maxPath = Math.max(...normalised.paths.map(l => l.map(p => p[2])).flat());

        normalised.nodes = normalised.nodes.map(l => l.map(n => (n / maxNode) * size));
        normalised.paths.forEach(layer => layer.forEach(path => path[2] = (path[2] / maxPath) * size / 4));


        const layerWidth = (ctx.canvas.width - 80) / (normalised.nodes.length + 1);
        const radius = 20; 

        let nodePositions = [];

        
        for (let layer = 0; layer < normalised.nodes.length; layer++) {
            const layerHeight = (ctx.canvas.height - 80) / (normalised.nodes[layer].length + 1);
            let currentLayerPositions = [];

            for (let node = 0; node < normalised.nodes[layer].length; node++) {
                const x = ((layer + 1) * layerWidth) + 40;
                const y = ((node + 1) * layerHeight) + 40;
                currentLayerPositions.push({ x, y });

                
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(x, y, Math.abs(normalised.nodes[layer][node]), 0, Math.PI * 2);
                ctx.fillStyle = "#3498db";
                ctx.fill();
                ctx.stroke();
            }

            nodePositions.push(currentLayerPositions);
        }

        
        ctx.strokeStyle = "#2c3e50";
        for (let layer = 0; layer < normalised.paths.length; layer++) {
            for (let path of normalised.paths[layer]) {
                const start = nodePositions[layer][path[0]];
                const end = nodePositions[layer + 1][path[1]];
                ctx.lineWidth = Math.abs(path[2]);

                ctx.beginPath();
                ctx.moveTo(start.x, start.y);
                ctx.lineTo(end.x, end.y);
                ctx.stroke();
            }
        }
    }

    
    train(inputs, expectedOutputs, learningRate = 0.01) {
        if (inputs.length !== expectedOutputs.length) {
            throw new Error("Inputs and expected outputs must have the same length.");
        }
        
        for (let i = 0; i < inputs.length; i++) {
            const output = this.run(inputs[i]);
            const target = expectedOutputs[i];

            
            const error = target.map((t, index) => t - output[index]);
            this.err += error.reduce((sum, e) => sum + Math.pow(e, 2), 0); 

            
            for (let layer = this.paths.length - 1; layer >= 0; layer--) {
                this.paths[layer].forEach(path => {
                    const inputValue = this.nodes[layer][path[0]];
                    const outputValue = this.nodes[layer + 1][path[1]];
                    const gradient = error[path[1]] * (this.act === Math.tanh ? 1 - Math.pow(outputValue, 2) : 1);
                    path[2] += learningRate * gradient * inputValue;
                });
            }
        }
        this.err /= inputs.length; 
    }
}

class GPUNet {
    constructor(network, actName, retName, populationSize, GPU = null, handletype = "common") {
        this.main = network;
        this.actTypes = ["sin", "cos", "tanh", "ReLu", "leakyReLu"];
        this.outTypes = ["sin", "cos", "tanh", "ReLu", "leakyReLu"];
        this.config = { handletype, populationSize, actName, retName, dataLength: 0 };
        this.gpu = GPU;
        this.population = Array.from({ length: populationSize }, () => this.main.clone());
    }

    setUp() {
        if (this.gpu !== null) {
            this.runKernel = this.gpu.createKernel(this.kernelFunction()).setOutput([this.config.populationSize, this.main.outputSize]);
        }
    }

    kernelFunction() {
        return function kernel() {
            function activate(value) {
                return value < 0 ? 0.1 * value : value;
            }
            function outputFunction(value) {
                return Math.tanh(value);
            }
            function netRun(net, inputs) {
                for (let i = 0; i < inputs.length; i++) {
                    net[0][0][i] = inputs[i];
                }

                for (let layerNum = 0; layerNum < net[1].length; layerNum++) {
                    let layer = net[1][layerNum];
                    for (let j = 0; j < layer.length; j++) {
                        let [fromNode, toNode, weight] = layer[j];
                        net[0][layerNum + 1][toNode] += net[0][layerNum][fromNode] * weight;
                    }

                    for (let k = 0; k < net[0][layerNum + 1].length; k++) {
                        net[0][layerNum + 1][k] = activate(net[0][layerNum + 1][k]);
                    }
                }

                return net[0][net[0].length - 1].map(outputFunction);
            }

            let input = this.constants.data[this.thread.x];
            let net = this.constants.population[this.thread.x];
            return netRun(net, input);
        };
    }

    run(inputs) {
        if (this.config.dataLength !== inputs.length) {
            this.config.dataLength = inputs.length;
            this.setUp();
        }
        this.runKernel.setConstants({ population: this.gpuPopulation(), data: inputs });
        return this.runKernel();
    }

    train(inputs, expectedOutput, errcalc) {
        this.config.dataLength = inputs.length;
        this.setUp();
        this.runKernel.setConstants({
            population: this.gpuPopulation(),
            data: inputs,
            train: true,
            output: expectedOutput
        });
        this.runKernel().forEach((data, i) => {
            this.population[i].err = errcalc(data);
        });
    }

    refillPopulation() {
        this.population = Array.from({ length: this.config.populationSize }, () => this.main.clone());
    }

    gpuPopulation() {
        return this.population.map(net => [net.nodes, net.paths]);
    }

    mutatePopulation(a, b) {
        this.population.forEach(net => {
            net.mutateNodes(a);
            net.mutatePaths(b);
        });
    }

    select(selector, callback) {
        if (selector === 'errorRate') {
            this.main = this.population.sort((a, b) => callback(a.err, b.err))[0].clone;
        }
    }

    get resultNet() {
        return this.main.clone();
    }

    get err() {
        return this.main.err;
    }
}


if (typeof self === "undefined") {
    module.exports = { NeuralNet, GPUNet };
}
