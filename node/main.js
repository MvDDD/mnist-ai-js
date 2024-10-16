let { NeuralNet } = require("./neural.js")
let fs = require("fs")
let GPU = (()=>{try{return require("gpu.js")}catch{return require("./gpu.js")}})();

let mnist = JSON.parse(fs.readFileSync("./mnist_DATA_SMALL.json"))


console.clear()

function logProgress(...array) {
    array.forEach((element, index) => {
        process.stdout.write(`\u001b[F`);
    });
    array.slice().forEach((element, index) => {
        process.stdout.write(element + "                    \n");
    });
}


let populationSize = 2

let net = new NeuralNet(
	[784, 10, 10],
	(a) => Math.max(a, 0),
	(a) => Math.tanh(a),
	random = Math.random
).fromString(fs.readFileSync("./model.json"))

let errRateDivider = 200
let errRate = mnist.length
let iterationsNoImprovement = 0


function err(rate) {
	return rate / errRateDivider
}

for (let i = 0; i < 100; i++) {

	let population = Array(populationSize).fill(0).map(a => {
		let nn = net.clone()
		nn.mutateNodes(err(errRate))
		nn.mutatePaths(err(errRate))
		return { nn, err: 0 }
	})

	// Evaluate each network in the population
	population.forEach((model, modelNr) => {
		mnist.forEach((image, imgNr) => {
			let out = model.nn.run(image.image.split(""))
			let selected = [null, -1]
			out.forEach((output, i) => {
				if (output > selected[0]) {
					selected[0] = output
					selected[1] = i
				}
			})
			if (imgNr%100==0)logProgress("iteration: " + i, "model: " + modelNr, "image: " + imgNr, "errRate (iteration): " + ((errRate/mnist.length)*100).toFixed(2) + "%")
			model.err += (image.label == selected[1] ? 0 : 1)
		})
	})
	population.sort((a, b) => a.err - b.err)
		iterationsNoImprovement = 0
	if (errRate > population[0].err){
	net = population[0].nn
	errRate = population[0].err
	} else {
		iterationsNoImprovement++
		if (iterationsNoImprovement > 3){
			errRateDivider += errRateDivider/20
		}
	}
}

// Save the best model to a JSON file
require("fs").writeFileSync("./model.json", net.toString())
