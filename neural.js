const NeuralNet = (() => {
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
                    this.max = layers.max || Infinity
                } else {
                    this.max = layers.max || Infinity
                    this.act = act;
                    this.out = out;
                    this.random = random;
                    this.nodes = layers.map(function(i) {
                        return Array(i).fill(0).map(() => 0);
                    });

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
            this.paths = this.paths.map(layer => layer.map(path => {
                let newWeight = path[2] + ((this.random() - 0.5) * amount);

                newWeight = Math.max(Math.min(newWeight, this.max), -this.max);
                return [path[0], path[1], newWeight];
            }));
        }


        mutateNodes(amount) {
            this.nodes = this.nodes.map(layer => layer.map(node => {
                let newNode = node + ((this.random() - 0.5) * amount);

                newNode = Math.max(Math.min(newNode, this.max), -this.max);
                return newNode;
            }));
        }

        clone() {
            let n = new NeuralNet({ nodes: this.nodes.map(l => l.slice()), paths: this.paths.map(l => l.map(p => p.slice())), act: this.act, out: this.out, err: this.err, max:this.max});
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
                out: this.out.toString(),
                max: this.max
            });
        }

        export () {
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

        fromFile(path) {
            try {
                return this.fromString(require("fs").readFileSync(path, 'utf8'));
            } catch (error) {
                console.error("Error reading from file:", error);
                return this;
            }
        }


        from(net) {
            return net.clone();
        }

        normalize() {
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
    }

    class GPUNet {
        constructor(net, populationSize, runtype, gpu) {
            // Helper function to convert string to named function
            function convertToNamedFunction(functionString, name) {
                functionString = functionString.trim();
                if (functionString.includes('=>')) {
                    let arrowParts = functionString.split('=>');
                    let params = arrowParts[0].trim();
                    let body = arrowParts[1].trim();
                    if (!params.startsWith('(')) {
                        params = '(' + params + ')';
                    }
                    if (!body.startsWith('{')) {
                        body = `{ return ${body}; }`;
                    }
                    return `function ${name}${params} ${body}`;
                }
                const functionPattern = /function\s*(\w*)\s*([^)]*)\s*{([\s\S]*)}/;
                const match = functionString.match(functionPattern);
                if (match) {
                    let params = match[2].trim();
                    let body = match[3].trim();
                    return `function ${name}(${params}) { ${body} }`;
                }
                throw new Error("Invalid function string");
            }

            function runNet(nodes, paths, input) {
                
                let nodes2 = [];
                for (let i = 0; i < nodes.length; i++){
                    nodes2[i] = nodes[i];
                }
                // Set the inputs
                for (let i = 0; i < this.constants.nodeOffsets[1]; i++) {
                    nodes2[i] += input[i];
                }

                // Prepare output array
                let out = [];

                // Run the network across layers
                for (let layer = 0; layer < this.constants.numLayers - 1; layer++) {
                    for (let path = 0; path < this.constants.pathOffsets[layer] / 3; path++) {
                        let p = pathIndex(layer, path);
                        let value = activation(nodes2[nodeIndex(layer, paths[p[0]])]);
                        nodes2[nodeIndex(layer + 1, paths[p[1]])] += value * paths[p[2]];
                    }
                }

                // Gather the output using direct indexing
                for (let i = 0; i < this.constants.outputSize; i++) {
                    out[i] = output(nodes2[nodeIndex(this.constants.numLayers - 1, i)]);
                }
                return out;
            }

            this.net = net
            this.runtype = runtype == "normal" ? 0 : 1;
            this.gpu = gpu;
            this.mainNodes = net.nodes.flat();
            this.mainPaths = net.paths.flat(2);
            this.nodeCumulativeSizes = this.computeCumulativeSizes(net.nodes.map(layer => layer.length));
            this.pathCumulativeLengths = this.computeCumulativeSizes(net.paths.map(layer => layer.length * 3));
            this.populationSize = populationSize;
            this.inputs = []

            // Prepare the kernel functions as string representations
            let nodeIndex = (function nodeIndex(layer, node) { return this.constants.nodeOffsets[layer] + node; }).toString();
            let pathIndex = (function pathIndex(layer, path) { const startIndex = this.constants.pathOffsets[layer] + (path * 3); return [startIndex, startIndex + 1, startIndex + 2]; }).toString();
            let activation = convertToNamedFunction(net.act.toString(), "activation");
            let output = convertToNamedFunction(net.out.toString(), "output");
            runNet = runNet.toString()
            console.log("init")
            this.funcStr = `function() {
                ${nodeIndex};
                ${pathIndex};
                ${activation};
                ${output};
                ${runNet};

                let outputArray = [];
                if (this.constants.runtype == 1) {
                    for (let i = 0; i < this.constants.dataSize; i++) {
                        outputArray[i] = runNet(this.constants.pNode[this.thread.x], this.constants.pPath[this.thread.x], this.constants.dataset[i]);
                    }
                    return outputArray; // Return the output array for population-based
                } else if (this.constants.runtype == 0) {
                    for (let i = 0; i < this.constants.populationSize; i++) {
                        outputArray[i] = runNet(this.constants.pNode[i], this.constants.pPath[i], this.constants.dataset[this.thread.x]);
                    }
                    return outputArray; // Return the output array for dataset-based
                }
            }`;
        }

        run(mutation){
            this.recompile(mutation)
            return this.kernel()
        }

        // Prepare the GPU kernel with constants and input data
        setInput(data) {
            this.inputs = data;
            console.log("create")
            this.kernel = this.gpu.createKernel(eval("(()=>{return " + this.funcStr + "})()"))
            this.recompile(0)
            console.log("compilation done")
        }

        // Recompile the GPU kernel after mutation
        recompile(mutation) {
            console.log("setoutput")
            this.kernel
            .setConstants(this.prepareConstants(mutation))
            console.log("constants done")

            let outputSize = [this.runtype ? this.populationSize : this.inputs.length]
            this.kernel
            .setOutput([outputSize])
            console.log("setoutput done")
        }

        // Prepare constants with mutation applied
        prepareConstants(mutation) {
            console.log("prepare inputs")
            const mf = () => ((Math.random() - 0.5) * mutation);

            const nodes = this.mainNodes.slice();
            const paths = this.mainPaths.slice();

            if (!Array.isArray(nodes) || !Array.isArray(paths)) {
                throw new Error("Nodes or paths are not initialized properly");
            }

            const netsNodes = Array(this.populationSize).fill(0).map(() => nodes.map(a => a + mf()));
            const netsPaths = Array(this.populationSize).fill(0).map(() => paths.map(p => [p[0], p[1], p[2] + mf()]));
            console.log("net")

            return {
                pNode: netsNodes,
                pPath: netsPaths,
                dataset: this.inputs,
                nodeOffsets: this.nodeCumulativeSizes,
                pathOffsets: this.pathCumulativeLengths,
                dataSize: this.inputs.length,
                runtype: this.runtype,
                populationSize:this.populationSize,
                numlayers:this.net.nodes.length,
                outputSize:this.net.nodes[this.net.nodes.length -1].length
            };
        }

        // Compute cumulative sizes for nodes and paths across layers
        computeCumulativeSizes(sizes) {
            const cumulative = [0];
            for (let i = 1; i < sizes.length; i++) {
                cumulative[i] = cumulative[i - 1] + sizes[i - 1];
            }
            return cumulative;
        }

        // Get node index from cumulative sizes
        getNodeIndex(layer, node) {
            return this.nodeCumulativeSizes[layer] + node;
        }

        // Get path index from cumulative sizes
        getPathIndex(layer, path) {
            const startIndex = this.pathCumulativeLengths[layer] + path * 3;
            return [startIndex, startIndex + 1, startIndex + 2];
        }
    }


    return { NeuralNet, GPUNet };
})();

module.exports = NeuralNet