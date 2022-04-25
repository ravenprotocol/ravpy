import ravop as R
import time


print("Testing all ops :")

f=['neg', 'pos', 'add', 'sub', 'exp', 'natlog', 'square', 'pow', 'square_root', 'cube_root', 
'abs', 'sum', 'sort', 'reverse', 'min', 'max', 'argmax', 'argmin', 'transpose', 'div', 'mul', 
'matmul', 'multiply', 'dot', 'unique', 'slice', 'greater', 'greater_equal', 
'less', 'less_equal', 'equal', 'not_equal', 'logical_and', 'logical_or', 
'logical_not', 'logical_xor', 'mean', 'average', 'mode', 'variance', 'median', 
'percentile', 'random', 'bincount', 'concat', 'cube']


algo = R.Graph(name='lin_reg', algorithm='linear_regression', approach='distributed')

a=R.Tensor([[1,2,3],[4,5,6],[7,8,9]])

tens=R.where(R.t([1,1,1,1]),R.t([2,2,2,2]),condition=[True,True,False,False])
print("where ->\n",tens())

q1=R.neg(a)
print("neg ->\n",q1())

q2=R.pos(a)
print("pos ->\n",q2())

q3=R.add(a,a)
print("add ->\n",q3())

q4=R.sub(a,a)
print("sub ->\n",q4())

q5=R.exp(a)
print("exp ->\n",q5())

q6=R.natlog(a)
print("natlog ->\n",q6())

q7=R.square(a)
print("square ->\n",q7())

q8=R.pow(a,a)
print("pow ->\n",q8())

q9=R.square_root(a)
print("square_root ->\n",q9())

q10=R.cube_root(a)
print("cube_root ->\n",q10())

q11=R.abs(a)
print("abs ->\n",q11())

q12=R.sum(a)
print("sum ->\n",q12())

q13=R.sort(a)
print("sort ->\n",q13())

q14=R.reverse(a)
print("reverse ->\n",q14())

q17=R.argmax(a)
print("argmax ->\n",q17())

q18=R.argmin(a)
print("argmin ->\n",q18())

q19=R.transpose(a)
print("transpose ->\n",q19())

q20=R.div(a,a)
print("div ->\n",q20())

q21=R.mul(a,a)
print("mul ->\n",q21())

q22=R.matmul(a,a)
print("matmul ->\n",q22())

q23=R.multiply(a,a)
print("multiply ->\n",q23())

q24=R.dot(a,a)
print("dot ->\n",q24())

q25=R.unique(a)
print("unique ->\n",q25())

q26=R.slice(a,begin=0,size=1)
print("slice ->\n",q26())

q27=R.greater(a,a)
print("greater ->\n",q27())

q28=R.greater_equal(a,a)
print("greater_equal ->\n",q28())

q29=R.less(a,a)
print("less ->\n",q29())

q30=R.less_equal(a,a)
print("less_equal ->\n",q30())

q31=R.equal(a,a)
print("equal ->\n",q31())

q32=R.not_equal(a,a)
print("not_equal ->\n",q32())

q33=R.logical_and(a,a)
print("logical_and ->\n",q33())

q34=R.logical_or(a,a)
print("logical_or ->\n",q34())

q36=R.logical_xor(a,a)
print("logical_xor ->\n",q36())

q37=R.mean(a)
print("mean ->\n",q37())

q38=R.average(a)
print("average ->\n",q38())


q40=R.variance(a)
print("variance =>",q40())

q45=R.cube(a)
print("cube ==>",q45())


arr1d=R.Tensor([1,2,3,4,5,6,7,8,8,9])
arr2d=R.Tensor([[1,2,3],[5,6,7],[9,10,11]])

a1=R.expand_dims(arr1d)
print("expand dims->",a1())

a2=R.inv(arr2d)
print(" inv ->",a2())

a3=R.gather(arr1d,R.Tensor([1,2,3,4]))
print(" gather ->",a3())

a4=R.stack(a)
print("stack ->",a4())



a6=R.find_indices(arr1d,R.Tensor([1,6,3]))
print("find_indices->",a6())

a7=R.shape(arr2d)
print("shape->",a7())

a8=R.mean(arr1d)
print("mean->",a8())

a9=R.average(arr1d)
print("average->",a9())



a11=R.variance(arr1d)
print("variance->",a11())


a13=R.std(arr1d)
print(" std ->",a13())


arr1d=R.Tensor([1,2,3,4,5,6,7,8,8,9])
a14=R.one_hot_encoding(arr1d,depth=4)
print("one_hot_encoding->",a14())



algo.end()