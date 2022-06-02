import numpy as np
import ravop.ravop as R
import json





def compute_locally_bm(*args, **kwargs):
    operator = kwargs.get("operator", None)
    op_type = kwargs.get("op_type", None)
    print("Operator", operator)
    if op_type == "unary":
        value1 = args[0]
        t1=time.time()
        eval("{}({})".format(numpy_functions[operator], value1))

    elif op_type == "binary":
        value1 = args[0]
        value2 = args[1]

        eval("{}({}, {})".format(numpy_functions[operator], value1, value2))


ravpyops=['neg', 'pos', 'add', 'sub', 'exp', 'natlog', 'square', 'pow', 'square_root', 'cube_root', 'abs', 'sum', 'sort', 'reverse', 'min', 'max', 'argmax', 'argmin', 'transpose', 'div']

unary2d=['square','cube_root','abs', 'sum','min', 'max', 'argmax', 'argmin','transpose','exp','sort',]
binary2dops=['add', 'sub','mul','pow',]
global v
v=[]
sizes=[5]
for i in sizes:
    arr=np.random.rand(i,i)
    for j in unary2d:
            string="global v;from ravsock.events.scheduler import create_payload ; v.append( create_payload(R."+j+"(R.Tensor("+str(arr.tolist())+" ))))"
            exec(string)

for i in sizes:
    arr=np.random.rand(i,i)
    for j in binary2dops:
            string="global v;from ravsock.events.scheduler import create_payload ; v.append( create_payload(R."+j+"(R.Tensor("+str(arr.tolist())+") ,R.t("+str(arr.tolist())+") )))"
            exec(string)


with open("benchmarkops_ravpy.json", "w") as outfile:
    json.dump(v, outfile)

