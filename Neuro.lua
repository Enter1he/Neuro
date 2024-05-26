-- Neuro.lua

local Neuro = {}

local random = math.random;
local sqrt =  math.sqrt;
local cos = math.cos;
local log = math.log;
local exp = math.exp;
local pi = math.pi;
local abs = math.abs;

function sigmoid(x)
    return 1/(1+exp(-x))
end

function square(x)
    return sigmoid(x*x)
end

function absolute(x)
    return sigmoid(x > 0 and x or -x)
end

function sine(x)
    return sigmoid(x > 0 and x or -x)
end

function linear(x)
    return x
end

function tanh(x)
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
end

function ReLU(x)
    return x < 0 and 0 or x
end

function Latch(x)
    return x == 0 and x or 0
end

function gaussian(mean)
    return sigmoid(((-2 * log(random()))^0.5)*cos(2 * pi * random()) + mean)
end

Neuro.funcs = {
    [1] = sigmoid;
    [2] = Latch;
    [3] = sine;
    [4] = tanh;
    [5] = ReLU;
    [6] = gaussian;
    [7] = linear;
}

local N_funcs = Neuro.funcs

Neuro.new = function(ins, hid, out)
    ins = ins or 0
    out = out or 0
    hid = hid or 0
    local N = {
        -- neurons
        neur = {};
        funcs = {};
        ins = 1; -- where inputs start
        hid = ins+out; -- where hidden starts
        out = ins; -- where outputs start
        nn = ins + hid + out;
        -- synapses
        syns = {};
        ion = {}; -- input and output neurons
        name = {};
        present = {};
        ns = 0; -- number of synapses
    }
    local neur = N.neur
    local funcs = N.funcs
    for i = 1, ins do
        neur[i] = 0.0
    end
    for i = N.out, ins+out do
        neur[i] = 0.0
    end
    for i = N.hid, ins+out+hid do
        neur[i] = 0.0
        funcs[i] = Neuro.funcs[5]
    end
    return N
end

Neuro.clone = function(N, cN)
    cN = cN or {}
    for k,v in pairs(N) do
        local vt = type(v)
        if vt == "table" then
            local a = {}
            cN[k] = a
            for i = 1, #v do
                a[i] = v[i]
            end
        else
            cN[k] = v
        end
    end
    return cN
end


local fmt_str = "n%dn%d"
local fmt = string.format
Neuro.addSynapse = function(N, n1, n2)
    if not n1 or not n2 then
        error"Invalid numbers!"
    end
    local name = fmt(fmt_str, n1, n2)
    local check = N.present[name]
    local syns = N.syns
    if check then
        syns[check] = syns[check] * 2
        return
    end
    N.ns = N.ns + 1
    local s = N.ns
    local ion = N.ion

    syns[s] = 0.0
    ion[s*2-1] = n1
    ion[s*2] = n2
    N.present[name] = s
    N.name[s] = name
end

Neuro.removeSynapse = function(N, s)
    local syns = N.syns
    local ion = N.ion
    local ns = N.ns
    local name = N.name
    
    syns[s], syns[ns] = syns[ns], syns[s]
    ion[s*2-1], ion[ns*2-1] = ion[ns*2-1], ion[s*2-1]
    ion[s*2], ion[ns*2] = ion[ns*2], ion[s*2]
    print(name[s], s)
    N.present[name[s]] = false
    name[s], name[ns] = name[ns], name[s]
    ns = ns - 1
    N.ns = ns
end

Neuro.addNeuron = function(N, s)
    
    N.nn = N.nn + 1
    local n = N.nn
    local syns = N.syns
    local ion = N.ion

    N.neur[n] = 0.0
    N.funcs[n] = N_funcs[math.random(1,6)]
    Neuro.addSynapse(N, n, ion[s*2])
    ion[s*2] = n
end

Neuro.count = function(N)
    local syns = N.syns
    local ion = N.ion
    local neur = N.neur
    local funcs = N.funcs
    for i = 1, N.ns do
        local n1, n2 = ion[i*2-1], ion[i*2]
        neur[n2] = neur[n2] + neur[n1]*syns[i]
    end
    for i = N.hid+1, N.nn do
        neur[i] = funcs[i](neur[i])
    end
    for i = N.out+1, N.ins-1 do
        neur[i] = funcs[i](neur[i])
    end
end



return Neuro