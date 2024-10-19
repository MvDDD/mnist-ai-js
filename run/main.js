let { NeuralNet } = require("./../neural.js");
let fs = require("fs");
let mnist = JSON.parse(fs.readFileSync("./../mnist_SMALL.json")).slice(-1000);

mnist = mnist.map(a => {
    return {
        image: a.image, // Assuming images are already preprocessed to a suitable format
        label: a.label
    };
});

let dataSize = mnist.length;

console.clear();

function logProgress(...array) {
    array.forEach((element, index) => {
        process.stdout.write(`\u001b[F`);
    });
    array.slice().forEach((element, index) => {
        process.stdout.write(element + "    \n");
    });
}


function err(rate) {
	return rate / errorRateDivider
}

let populationSize = 20;
let iterations = 100;

let net = new NeuralNet(
    [784, 10, 10],
    (a) => Math.tanh(Math.max(a, a / 10)),
    (a) => Math.tanh(a),
    random = Math.random
).fromFile("./../model.json");
net.max = 255;
net.act = (a) => Math.tanh(Math.max(a, 0))

let errorRateDivider = 20000;
let errorRate = 1000;
let currentLoss = 5000;
let iterationsNoImprovement = 0;

let i = 0;
while (true){//(((errorRate / mnist.length) * 100) > 50) {
    i++;
    let population = Array.from({ length: populationSize }, () => {
        let childNet = net.clone();
        childNet.mutateNodes(err(currentLoss)); // Ensure 'err' is used if needed
        childNet.mutatePaths(err(currentLoss)); // Ensure 'err' is used if needed
        return { net: childNet, corrects: 0, loss: 0 }; // Removed err since it's not used
    });

    // Evaluate each network in the population
    population.forEach((model, modelNr) => {
        mnist.forEach((image, imgNr) => {
            let out = model.net.run(image.image);
            let selected = [0, -1]; // [bestOutput, bestIndex]

            // Identify the predicted label
            out.forEach((output, i) => {
                if (output > selected[0]) {
                    selected[0] = output; // Store best output value
                    selected[1] = i;      // Store the index of the best output (predicted class)
                }
            });

            // Update correct predictions and losses
            if (image.label === selected[1]) {
                model.corrects++; // Increase correct count if prediction is right
            } else {
                model.loss += 1; // Increment loss for incorrect predictions
            }

            if ((imgNr + 1) % 100 === 0) {
                logProgress(
                    "iteration: " + (i + 1),
                    "model: " + (modelNr + 1) + " of " + populationSize,
                    "image: " + (imgNr + 1) + " of " + dataSize,
                    "errorRate (iteration): " + ((errorRate / mnist.length) * 100).toFixed(10) + "%",
                    "divider: " + errorRateDivider.toFixed(5),
                    "loss: " + model.loss, // Log current model loss
                    "output: ",
                    ...(out.map(a => "    " + a + "                "))
                );
            }
        });
    });

    // Sort population by their performance metrics
    population.sort((a, b) => a.loss - b.loss);
    
    iterationsNoImprovement = 0;
    if (currentLoss > population[0].loss) {
        net = population[0].net.clone();
        errorRate = population[0].corrects; // Update error rate based on correct predictions
        currentLoss = population[0].loss; // Update the current loss
    }
    if (errorRate < population.corrects){
        iterationsNoImprovement++;
        if (iterationsNoImprovement > 3) {
            errorRateDivider -= (errorRateDivider + 1) ** 0.5; // Increase difficulty after several iterations without improvement
        }
    }

    if (i % 10 === 0) {
        fs.writeFileSync("./../model.json", net.toString());
    }
}
fs.writeFileSync("./../model.json", net.toString());
