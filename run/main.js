let { NeuralNet, GPUNet } = require("./neural.js")
let fs = require("fs")
let GPU = (()=>{try{return require("gpu.js")}catch{console.warn("falling back on cpu: no gpujs found"); return require("./gpu.js")}})();

let mnist = JSON.parse(fs.readFileSync("./mnist_DATA_SMALL.json")).slice(-100)
mnist = mnist.map(a=>{
	return {
		image:a.image.split("").map(a=>a=="0"?0:1),
		label:a.label
	}
})
let dataSize = mnist.length

console.clear()

function logProgress(...array) {
    array.forEach((element, index) => {
        process.stdout.write(`\u001b[F`);
    });
    array.slice().forEach((element, index) => {
        process.stdout.write(element + "    \n");
    });
}


let populationSize = 10
let iterations = 100

let net = new NeuralNet(
	[784, 10, 10, 10, 10, 10],
	(a) => Math.max(a, 0),
	(a) => Math.tanh(a),
	random = Math.random
).fromFile("./model.json").normalize()

let errorRateDivider = 200
let errorRate = mnist.length
let currentLoss = 0
let iterationsNoImprovement = 0


function err(rate) {
	return rate / errorRateDivider
}

let i = 0;
//while (((errorRate/mnist.length)*100) > 50) {
//i++
	let population = Array(populationSize).fill(0).map(a => {
		let childNet = net.clone()
		childNet.mutateNodes(err(errorRate))
		childNet.mutatePaths(err(errorRate))
		return { net:childNet, corrects: 0, loss:0 }
	})

	// Evaluate each network in the population
	population.forEach((model, modelNr) => {
		mnist.forEach((image, imgNr) => {
			let out = model.net.run(image.image)
			let selected = [null, -1]
			out.forEach((output, i) => {
				if (output > selected[0]) {
					selected[0] = output
					selected[1] = i
				}
			})
			if ((imgNr+1)%100==0)logProgress("iteration: " + (i+1), "model: " + (modelNr+1) + " of " + populationSize, "image: " + (imgNr+1) + " of " + dataSize, "errorRate (iteration): " + ((errorRate/mnist.length)*100).toFixed(10) + "%", "divider:" + errorRateDivider.toFixed(5))
			model.err += (image.label == selected[1] ? 0 : 1)
		})
	})
	population.sort((a, b) => a.err - b.err)
		iterationsNoImprovement = 0
	if (currentLoss > population[0].loss){
	net = population[0].nn
	errorRate = population[0].err
	} else {
		iterationsNoImprovement++
		if (iterationsNoImprovement > 3){
			errorRateDivider = (errorRateDivider+1) ** 2
		}
	}
	//errorRateDivider *= 0.99
	if (i%100 == 0){
		//fs.writeFileSync("./model.json", net.toString())
	}
//}
//fs.writeFileSync("./model.json", net.toString())
