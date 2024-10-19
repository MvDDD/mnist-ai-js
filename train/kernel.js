function() {
                function nodeIndex(layer, node) { return this.constants.nodeOffsets[layer] + node; };
                function pathIndex(layer, path) { const startIndex = this.constants.pathOffsets[layer] + (path * 3); return [startIndex, startIndex + 1, startIndex + 2]; };
                function activation(a) { return Math.tanh(Math.max(a, 0)); };
                function output(a) { return Math.tanh(a); };
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
            };

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
            }