let {NeuralNet, GPUNet} = require("./../neural.js")
let fs = require("fs")
let {GPU} = require("gpu.js")

let dataset = JSON.parse(fs.readFileSync("./../mnist_SMALL.json")).slice(-5)

let images = dataset.map(a=>a.image)
let labels = dataset.map(a=>a.label)

let net = new NeuralNet(
	[
		dataset[0].image.length,
		10,
		10
		]
	).fromFile("./../model.json")

let gpu = new GPU()
let gpuNet = new GPUNet(net, 10, "other", gpu)


gpuNet.setInput(images)

fs.writeFileSync("kernel.js", gpuNet.funcStr)