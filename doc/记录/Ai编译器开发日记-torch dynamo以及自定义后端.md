# torch dynamo以及自定义后端

torch compiler 由两部分组成，编译器的前端 Torch Dynamo 和 后端 Induactor。
torch compiler 的论文：https://pytorch.org/assets/pytorch2-2.pdf

## Torch Dynamo

### 捕获fx_graph

torch dynamo的作用是用来解析捕获激活图的。同时做一些常量折叠的优化。

dynamo会将torch的Module 转换为fx_graph 图。先大致说一下流程（如有偏差，以torch官方的说法为准~~）。torch 会利用FakeTensor执行一遍nn.Module，在这个时候，dynamo会捕获python解释器调用函数的接口，获取Module当中实际调用的函数。

根据截取的函数调用参数等信息，构建出fx_graph图。构建fx_graph的机制是和dispatch机制绑在一起。

构建出的fx_graph 长这个样子。

```
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

```
opcode         name           target                                                   args              kwargs
-------------  -------------  -------------------------------------------------------  ----------------  --------
placeholder    s0             s0                                                       ()                {}
placeholder    l_x_           L_x_                                                     ()                {}
call_module    l__self___fc1  L__self___fc1                                            (l_x_,)           {}
call_function  x              <built-in method relu of type object at 0x7fd5543f9500>  (l__self___fc1,)  {}
call_module    l__self___fc2  L__self___fc2                                            (x,)              {}
call_function  x_1            <built-in method relu of type object at 0x7fd5543f9500>  (l__self___fc2,)  {}
call_module    x_2            L__self___fc3                                            (x_1,)            {}
output         output         output                                                   ((x_2,),)         {}
```

fx_graph 由多个fx.node组成，node有若干个类型：

- placeholder ： 模型的输入，一部分是动态的dim，一部分是tensor的输入。
- call_module： 调用了一个nn.Module,上文的例子就是调用了nn.Linear。
- call_function:   调用了一个函数。就是dispatch实际注册的算子函数名。
- call_method：调用的是python的内建函数。
- out put：输出。

拿到这张图的时候，其实torch就已经做了一些简单的常量折叠等优化了。

但是这种图的call_module可能是开发者自定义的nn.Module，在torch的aot_module_simplified函数内部可以把所有的call_module变成call_function和call_method。
然后fx图就变成了这样：

```
placeholder    primals_1  primals_1           ()                                                          {}
placeholder    primals_2  primals_2           ()                                                          {}
placeholder    primals_3  primals_3           ()                                                          {}
placeholder    primals_4  primals_4           ()                                                          {}
placeholder    primals_5  primals_5           ()                                                          {}
placeholder    primals_6  primals_6           ()                                                          {}
placeholder    primals_7  primals_7           ()                                                          {}
placeholder    primals_8  primals_8           ()                                                          {}
call_function  t          aten.t.default      (primals_1,)                                                {}
call_function  addmm      aten.addmm.default  (primals_2, primals_8, t)                                   {}
call_function  relu       aten.relu.default   (addmm,)                                                    {}
call_function  t_1        aten.t.default      (primals_3,)                                                {}
call_function  addmm_1    aten.addmm.default  (primals_4, relu, t_1)                                      {}
call_function  relu_1     aten.relu.default   (addmm_1,)                                                  {}
call_function  t_2        aten.t.default      (primals_5,)                                                {}
call_function  addmm_2    aten.addmm.default  (primals_6, relu_1, t_2)                                    {}
output         output     output              ([addmm_2, primals_8, relu, t_1, relu_1, t_2, primals_7],)  {}
opcode         name                  target                           args                 
```

可以看到模型的输出有很多个，这是因为dynamo会自动生成反向传播的图，有些参数是需要返给反向的计算图进行反向传播的。

### 多张子图的由来

训练模式下至少会生成至少两张fx图，一张正向的、一张反向的。这是由partition_fn 来做切分的。如果模型很大或者模型里面有一些奇怪的操作，会把图切开。有可能一个模型生成很多个子图。

比如以下的示例：

```python
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(input_dim, 12)

    def forward(self, x):
        x = self.fc3(x)
        print("a")
        x = x + x
        x = x + x
        return x
```

dynamo 捕获的图会在print("a")的时候打断，导致获取两张子图。

```
-------------  ---------  ---------------  ------------------------------  --------
placeholder    primals_1  primals_1           ()                                {}
placeholder    primals_2  primals_2           ()                                {}
placeholder    primals_3  primals_3           ()                                {}
placeholder    primals_4  primals_4           ()                                {}
call_function  t          aten.t.default      (primals_1,)                      {}
call_function  addmm      aten.addmm.default  (primals_2, primals_4, t)         {}
output         output     output              ([addmm, primals_4, primals_3],)  {}
opcode         name        target                args                        kwargs
-------------  ----------  --------------------  --------------------------  --------
placeholder    primals_3   primals_3             ()                          {}
placeholder    primals_4   primals_4             ()                          {}
placeholder    tangents_1  tangents_1            ()                          {}
call_function  t_1         aten.t.default        (tangents_1,)               {}
call_function  mm          aten.mm.default       (t_1, primals_4)            {}
call_function  t_2         aten.t.default        (mm,)                       {}
call_function  sum_1       aten.sum.dim_IntList  (tangents_1, [0], True)     {}
call_function  view        aten.view.default     (sum_1, [12])               {}
call_function  t_3         aten.t.default        (t_2,)                      {}
output         output      output                ([t_3, view, None, None],)  {}
a
opcode         name       target           args                              kwargs
-------------  ---------  ---------------  --------------------------------  --------
placeholder    primals_1  primals_1        ()                                {}
placeholder    primals_2  primals_2        ()                                {}
placeholder    primals_3  primals_3        ()                                {}
call_function  add        aten.add.Tensor  (primals_3, primals_3)            {}
call_function  add_1      aten.add.Tensor  (add, add)                        {}
output         output     output           ([add_1, primals_1, primals_2],)  {}
opcode         name        target           args                      kwargs
-------------  ----------  ---------------  ------------------------  --------
placeholder    primals_1   primals_1        ()                        {}
placeholder    primals_2   primals_2        ()                        {}
placeholder    tangents_1  tangents_1       ()                        {}
call_function  add_2       aten.add.Tensor  (tangents_1, tangents_1)  {}
call_function  add_3       aten.add.Tensor  (add_2, add_2)            {}
output         output      output           ([None, None, add_3],)    {}
```

这个时候拿到了四张子图。

### decompositions

decomposition 参数可以将fx_garph 中比较复杂的target分解为比较简单的target组合，比如我们觉得：

```
placeholder    primals_1  primals_1           ()                                {}
placeholder    primals_2  primals_2           ()                                {}
placeholder    primals_3  primals_3           ()                                {}
placeholder    primals_4  primals_4           ()                                {}
call_function  t          aten.t.default      (primals_1,)                      {}
call_function  addmm      aten.addmm.default  (primals_2, primals_4, t)         {}
output         output     output              ([addmm, primals_4, primals_3],)  {}
```

aten.addmm.default 这个target太复杂了,可以 decomposition 上注册：

```
aten = torch.ops.aten
default_decompositions = {aten.addmm}
modle = SimpleNN(2)
modle_opt = torch.compile(
    model=modle,
    backend=aot_autograd(
        fw_compiler=compiler_demo_inner,
        decompositions=get_decompositions(default_decompositions),
    ),
    dynamic=True,
    fullgraph=False,
)
```

之后获得的fx 图就吧注册的算子分解成了其他的小算子：

```
placeholder    primals_1  primals_1        ()                              {}
placeholder    primals_2  primals_2        ()                              {}
placeholder    primals_3  primals_3        ()                              {}
placeholder    primals_4  primals_4        ()                              {}
call_function  t          aten.t.default   (primals_1,)                    {}
call_function  mm         aten.mm.default  (primals_4, t)                  {}
call_function  mul        aten.mul.Tensor  (mm, 1)                         {}
call_function  mul_1      aten.mul.Tensor  (primals_2, 1)                  {}
call_function  add        aten.add.Tensor  (mul, mul_1)                    {}
output         output     output           ([add, primals_4, primals_3],)  {}
```

### fx_graph 运行

在实际运行是，torch会逐个会对fx_graph 的结点逐个调用，调用的对象就是target。
这里我们将图中所有的aten.add.Tensor 替换我我们自己定义的 callable[只是简单的示例 ]：
调用函数的话,会发现在运行执行的是我们自己定义的call able (这里可以作为一个作为编译执行的入口)。

```python
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(input_dim, 12)

    def forward(self, x):
        # x = self.fc3(x)
        # print("a")
        x = x + x
        x = x + x
        return x

def test_add(lhs, rhs):
    print("call test add")
    return lhs + rhs


def compiler_demo_inner(gm: torch.fx.GraphModule, inputs):
    gm.print_readable(False)
    gm.graph.print_tabular()
    for node in gm.graph.nodes:
        if node.target == aten.add.Tensor:
            node.target = test_add

    return gm.forward

aten = torch.ops.aten
default_decompositions = {aten.addmm}
modle = SimpleNN(2)
modle_opt = torch.compile(
    model=modle,
    backend=aot_autograd(
        fw_compiler=compiler_demo_inner,
        decompositions=get_decompositions(default_decompositions),
    ),
    dynamic=True,
    fullgraph=False,
)
inputs = torch.ones(10, 2)
print( modle_opt(inputs))
```

```
call test add
call test add
tensor([[4., 4.],
        [4., 4.],
        [4., 4.],
        [4., 4.],
        [4., 4.],
        [4., 4.],
        [4., 4.],
        [4., 4.],
        [4., 4.],
        [4., 4.]])
```

fx_graph 的机制关于执行比较重要的就这些吧，在示例compiler_demo_inner中就可以写实际后端的编译逻辑了。
