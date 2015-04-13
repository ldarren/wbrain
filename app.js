// ref: http://www.cs.bham.ac.uk/~jxb/NN/nn.html
// dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
const
E = Math.E,
Pow = Math.pow,
fdata = fs.readFileSync('./data/bezdekIris.data')

var
fs = require('fs'),
inputs = [],
targets = [],
sigmoid = function(i){ return 1/(1+Pow(E, -i)) },
derivative = function(s){ return s*(1-s) },
activate = function(ins, weights){
	var out = weights[0]
	for (var i=1,l=weights.length; i<l; i++){
		out+=ins[i]*weights[i]
	}
	return sigmoid(out)
},
input = 0

!function(){
	// read file
	var lines = fdata.split('\n')
}

console.log('sigmoid', sigmoid(input))
console.log('sigmoid derivative', dSigmoid(input))


