// Ref: 
// http://www.cs.bham.ac.uk/~jxb/NN/nn.html
// http://visualstudiomagazine.com/Articles/2013/09/01/Neural-Network-Training-Using-Back-Propagation.aspx?p=1
// http://visualstudiomagazine.com/Articles/2013/12/01/Neural-Network-Training-Using-Particle-Swarm-Optimization.aspx?p=1
const
E = Math.E,
Pow = Math.pow,
Random = Math.random,
Floor = Math.floor,
Log = Math.log

var
sigmoid = function(i){ return 1/(1+Pow(E, -i)) },
derivative = function(s){ return s*(1-s) },
wrand = function(s){ return 2 * (Random()-0.5) * s},
sse = function(d){return 0.5*d*d}, // sum square error, d = target-output
cee = function(t, o){return t*Log(o)+((1-t)*Log(1-o))}, // cross entropy error
dSSE = function(t, o){return (t-o)*o*(1-o)},
dCEE = function(t, o){return t-o},
NeuralNetwork = function(arg){
    if ('object' !== typeof arg){
        throw 'wrong parameter: '+JSON.stringify(arguments)
    }else if (arg.length){
        this.reboot.apply(this, arguments)
    }else{
        this.recall.apply(this, arguments)
    }
},
backPropagate = function(output, target, layerSize, layers, dLayers, weights){
    var
    e = 0,
    t,o,
    j,jl,k,kl,m,ml,
    dlj,dli,li,wi,wik
    
    dlj=dLayers[layerSize.length-1]
    for(k=0,kl=target.length; k<kl; k++){
        t=target[k]
        o=output[k]
        e += sse(t-o)
        dlj[k+1]=dSSE(t, o)
    }

    for(j=layerSize.length-1; j; j--){
        ml=layerSize[j-1]
        kl=layerSize[j]
        dlj=dLayers[j]
        dli=dLayers[j-1]
        li=layers[j-1]
        wi=weights[j-1]

        for(m=1; m<ml; m++){ // lower 
            t=0

            for(k=1; k<kl; k++){ // upper 
                wik=wi[k]
                t+=wik[m]*dlj[k]
            }
            dli[m]=t*derivative(li[m])
        }
    }
console.log(e, JSON.stringify(target),JSON.stringify(output))
    return e
},
updateWeights = function(eta, alpha, layerSize, layers, dLayers, weights, dWeights){
    var k,kl,m,ml,wj,dwj,lj,dlj,dwjk,wjk
    for(var j=0,jl=layerSize.length-1; j<jl; j++){
        kl=layerSize[j+1]
        ml=layerSize[j]
        dwj=dWeights[j]
        wj=weights[j]
        dlj=dLayers[j+1]
        lj=layers[j]
        for(k=1; k<kl; k++){ // upper
            dwjk=dwj[k]
            wjk=wj[k]

            for(m=0; m<ml; m++){ // lower
                dwjk[m] = eta*lj[m]*dlj[k] + alpha*dwjk[m]
                wjk[m]+=dwjk[m]
            }
        }
    }
}

NeuralNetwork.prototype = {
    reboot: function(size, config){
        config = config || {}

        var
        layerSize = new Array(size.length),
        layers = [],
        dLayers = [],
        dWeights = [],
        weights = [],
        smallwt = config.smallwt || 0.5,
        i,l,j,jl,k,kl,w,dw,wj,dwj

        // initialize layer size and layers
        for(i=0,l=size.length; i<l; i++){
            jl = size[i]+1
            layerSize[i] = jl
            layers.push(new Array(jl))
            dLayers.push(new Array(jl))
        }

        // initialize weights and dweights
        for(i=0,l=layerSize.length-1; i<l; i++){
            jl=layerSize[i+1]
            kl=layerSize[i]
            weights.push(new Array(jl))
            dWeights.push(new Array(jl))
            w = weights[i]
            dw = dWeights[i]
            for(j=1; j<jl; j++){ // start from 1 because the first node's value is always 1
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

        this.layerSize = layerSize.slice()
        this.layers = layers
        this.dLayers = dLayers
        this.weights = weights
        this.dWeights = dWeights
    },
    learn: function(epochs, inputs, targets, config){
        var
        ranpat = [],
        layerSize = this.layerSize,
        layers = this.layers,
        dLayers = this.dLayers,
        weights = this.weights,
        dWeights = this.dWeights,
        eta = config.eta || 0.5,
        alpha = config.alpha || 0.9,
        idx,error,
        i,l,p

        for(i=0,l=inputs.length; i<l; i++) ranpat[i]=i

        for(var e=0; e<epochs; e++){
            for(l=inputs.length; l; l--, i=Floor(Random()*l), p=ranpat[l], ranpat[l]=ranpat[i], ranpat[i]=p);
            error=0

            for(i=0,l=ranpat.length; i<l; i++){
                idx=ranpat[i]
                
                error += backPropagate(this.think(inputs[idx]), targets[idx], layerSize, layers, dLayers, weights)
                updateWeights(eta, alpha, layerSize, layers, dLayers, weights, dWeights)                
            }
            if (error/l < 0.0004) return error
        }
        return error
    },
    think: function(input, filter){
        var
        layerSize = this.layerSize,
        layers = this.layers,
        weights = this.weights,
        li,lo,loo,wi,wj,
        i,l,j,jl,k,kl

        li = layers[0]
        li[0]=1
        for(i=0,l=input.length; i<l; i++) li[i+1] = filter ? filter[i]*input[i] : input[i]

        for(i=0,l=layerSize.length-1; i<l; i++){
            jl = layerSize[i+1]
            kl = layerSize[i]
            li=layers[i]
            lo=layers[i+1]
            wi=weights[i]
            lo[0]=1//first node is always 1
            for(j=1; j<jl; j++){
                wj=wi[j]
                loo=0
                for(k=0; k<kl; k++){
                    loo+=wj[k]*li[k]
                }
                lo[j] = sigmoid(loo)
            }
        }
        return layers[layers.length-1].slice(1)
    },
    memory: function(){
        return {
            layerSize: this.layerSize,
            layers: this.layers,
            dLayers: this.dLayers,
            weights: this.weights,
            dWeights: this.dWeights
        }
    },
    recall: function(memory){
        this.layerSize = memory.layerSize
        this.layers = memory.layers
        this.dLayers = memory.dLayers
        this.weights = memory.weights
        this.dWeights = memory.dWeights
    }
}

module.exports = NeuralNetwork
