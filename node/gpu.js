// ./gpu.js
class CPUKernel {
    constructor(fn) {
        this.fn = fn;
        this.outputDimensions = null;
        this.constants = {};
    }

    setOutput(dimensions) {
        this.outputDimensions = dimensions;
        return this;
    }

    setConstants(constants) {
        this.constants = constants;
        return this;
    }

    run(...args) {
        const output = this.createOutputArray();
        const inputArrays = this.prepareInputArrays(args);

        for (let i = 0; i < output.length; i++) {
            // Setting the context for the current thread
            this.thread = { x: i };
            // Execute the kernel function
            output[i] = this.fn(...inputArrays[i]);
        }

        return output;
    }

    createOutputArray() {
        const size = this.outputDimensions.reduce((a, b) => a * b, 1);
        return new Array(size);
    }

    prepareInputArrays(args) {
        // Flatten the input arrays based on output size
        const inputArrays = [];
        const size = this.outputDimensions.reduce((a, b) => a * b, 1);
        for (let i = 0; i < size; i++) {
            inputArrays.push(args);
        }
        return inputArrays;
    }
}

class GPU {
    constructor() {
        // This can be extended for additional configurations
    }

    createKernel(fn) {
        return new CPUKernel(fn);
    }
}

// Export the GPU class
module.exports = {GPU};
