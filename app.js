// ref: http://www.cs.bham.ac.uk/~jxb/NN/nn.html
// dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
const
E = Math.E,
Pow = Math.pow,
Random = Math.random,
Floor = Math.floor,
IRIS_CLASS = {'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}

var
fs = require('fs'),
path = require('path'),
sigmoid = function(i){ return 1/(1+Pow(E, -i)) },
derivative = function(s){ return s*(1-s) },
run = function(ins, weights){
	var out = weights[0]
	for (var i=1,l=weights.length; i<l; i++){
		out+=ins[i]*weights[i]
	}
	return sigmoid(out)
},
toNum = function(i){ return parseFloat(i, 10) },
readDataSet = function(fpath, inputs, targets){
    inputs.length = 0
    targets.length = 0
    var data = fs.readFileSync(path.isAbsolute(fpath) ? fpath : path.resolve(__dirname+path.sep+fpath), {encoding:'utf8'})
    if (!data) return 1
	var lines = data.split('\n')
    if (!lines || !lines.length) return 2

    for(var i=0,l=lines.length,arr; i<l; i++){
        arr = lines[i].split(',')
        if (!arr || 2>arr.length) break
        targets.push([0, IRIS_CLASS[arr.splice(-1, 1)[0]]])
        arr = arr.map(toNum)
        arr.unshift(0)
        inputs.push(arr)
    }
    return 0
},
wrand = function(smallwt){
    return 2 * (Random()-0.5) * smallwt
}

function NeuronNetwork(layerSize, config){
    config = config || {}

    var
    numInput = layerSize[0],
    numOutput = layerSize[layerSize.length-1],
    layers = [],
    dWeights = [],
    weights = [],
    smallwt = config.smallwt || 0.5,
    i,l,j,jl,k,kl,w,dw,wj,dwj

    layers.push(0)
    layers.push(new Array(numInput))

    for(i=1,l=layerSize.length-1; i<l; i++){
        jl = layerSize[i]+1
        layers.push(new Array(jl))
        layers.push(new Array(jl))
    }

    layers.push(new Array(numOutput))

    for(i=0,l=layerSize.length-1; i<l; i++){
        jl=layerSize[i+1]
        kl=layerSize[i]
        weights.push(new Array(jl))
        dWeights.push(new Array(jl))
        w = weights[i]
        dw = dWeights[i]
        for(j=1; j<jl; j++){
            w[j] = new Array(kl)
            dw[j] = new Array(kl)
            wj = w[j]
            dwj = dw[j]
            for(k=0; k<kl; k++){
                wj[k]=wrand(smallwt)
                dwj[k]=0
            }
        }
    }

    this.eta = config.eta || 0.5
    this.alpha = config.alpha || 0.9
    this.layers = layers
    this.layerSize = layerSize.slice()
    this.weights = weights
    this.dWeights = dWeights
}

NeuronNetwork.prototype = {
    train: function(loop, inputs, targets){
        var
        ranpat = [],
        layerSize = this.layerSize,
        error,
        i,l

        for(i=0,l=inputs.length; i<l; i++) ranpat[i]=i

        for(var epoch=0; epoch<loop; epoch++){
            for(l=inputs.length-1,i; l>-1; i=Floor(Random()*l), p=ranpat[l], ranpat[l]=ranpat[i], ranpat[i]=p, l--);
            error=0

            for(i=0,l=ranpat.length; i<l; i++){
                this.compute(inputs[ranpat[i]])
            }
        }
    },
    compute: function(inputs){
        var
        layers = this.layers,
        layerSize = this.layerSize,
        weights = this.weights,
        outputs = [],
        li,wi,lj,
        i,l,j,jl,k,kl

        for(i=0,l=layerSize.length-1; i<l; i++){
            jl = layerSize[i+1]
            kl = layerSize[i]
            li=layers[1+(i*2)]
            wi=weights[i]
            for(j=1; j<jl; j++){
                li[j]=wi[0]
                for(k=1; k<kl; k++){
                    li[j]+=wi[k]*
                }
            }
        }
    },
    log: function(){
        console.log(JSON.stringify(this.inputs))
        console.log(JSON.stringify(this.targets))
        console.log(JSON.stringify(this.layers))
        console.log(JSON.stringify(this.weights))
        console.log(JSON.stringify(this.dWeights))
    }
}

var inputs = [], targets = []
if (readDataSet('data/bezdekIris.data', inputs, targets)) throw 'invalid input files'

var nn = new NeuronNetwork([inputs[0].length, 2, targets[0].length])
nn.train(100000, inputs, targets)
