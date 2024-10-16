let fs = require("fs")

let mnist = JSON.parse(fs.readFileSync(process.argv[2]))
mnist = mnist.map(i => {
	return {image: i.image.map(val=>val>128?1:0).join(""), label:i.label}
})

fs.writeFileSync(process.argv[3], JSON.stringify(mnist))