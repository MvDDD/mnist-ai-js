class GPUNet {
	constructor(net, populationSize, runtype, gpu) {


		function convertToNamedFunction(functionString, name) {
			functionString = functionString.trim();
			if (functionString.includes('=>')) {
				let arrowParts = functionString.split('=>');
				let params = arrowParts[0].trim();t
				let body = arrowParts[1].trim();
				if (!params.startsWith('(')) {
					params = '(' + params + ')';
				}
				if (!body.startsWith('{')) {
					body = `{ return ${body}; }`;
				}
				return `function ${name}${params} ${body}`;
			}
			const functionPattern = /function\s*(\w*)\s*\(([^)]*)\)\s*{([\s\S]*)}/;
			const match = functionString.match(functionPattern);
			if (match) {
				let params = match[2].trim();
				let body = match[3].trim();
				return `function ${name}(${params}) { ${body} }`;
			}
			throw new Error("Invalid function string");
		}


		this.runtype = runtype
		this.gpu = gpu;
		this.mainNodes = net.nodes.flat();
		this.mainPaths = net.paths.flat(2);
		this.nodeCumulativeSizes = this.computeCumulativeSizes(net.nodes.map(layer => layer.size));
        this.pathCumulativeLengths = this.computeCumulativeSizes(net.paths.map(layer => layer.length * 3));  // 3 elements per path
        
        let nodeIndex = (function nodeIndex(layer, node) {return this.constants.nodeCumulativeSizes[layer] + node;}).toString()
        let pathIndex = (function pathIndex(layer, path){const startIndex = this.pathCumulativeLengths[layer] + path * 3;return [startIndex,startIndex+1,startIndex+2];}).toString()
        let activation = convertToNamedFunction(net.act.toString(), "activation")
        let output = 	convertToNamedFunction(net.out.toString(), "output")
        this.funcStr = `function(){${nodeIndex};${pathIndex};${activation};${output};
        if (this.constants.runtype == 0){//handle runtype normal
        	let output = []
        	for (let i = 0; i < this.constants.populationSize; i++){
        		output[i] = runNet()
        	}
        }
    }`
    }
    run(data){

    }

    setInput(data){
    	this.inputs = data
    	this.kernel = this.gpu.createKernel(eval("(()=>{return " + this.funcStr + "})()")).setOutput((this.runtype?this.population.length:this.data.length))
    	this.kernel
    }
    computeCumulativeSizes(sizes) {
    	const cumulative = [0];
    	for (let i = 1; i < sizes.length; i++) {
    		cumulative[i] = cumulative[i - 1] + sizes[i - 1];
    	}
    	return cumulative;
    }
    getNodeIndex(layer, node) {
    	return this.nodeCumulativeSizes[layer] + node;
    }
    getPathIndex(layer, path) {

    }
}
