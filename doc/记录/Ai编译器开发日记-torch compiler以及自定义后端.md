# torch compiler以及自定义后端

torch compiler 由两部分组成，编译器的前端 Torch Dynamo 和 后端 Induactor。

## Torch Dynamo

torch dynamo的作用是用来解析捕获激活图的。同时做一些常量折叠的优化。

dynamo会将torch的Module 转换为fx_graph 图。先大致说一下流程（如有偏差，以torch官方的说法为准~~）。torch 会利用FakeTensor执行一遍nn.Module，在这个时候，dynamo会捕获python解释器调用函数的接口，获取Module当中实际调用的函数。

根据截取的函数调用参数等信息，构建出fx_graph图。构建fx_graph的机制是和dispatch机制绑在一起（反正就是很复杂同事总是在吐槽这已经堆成屎山了，笑死）。

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

但是这种图的call_module可能是开发者自定义的nn.Module，因此torch还有retrace的功能可以把call_module变成call_function和call_method。
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
训练模式下会生成至少两张fx图，一张正向的、一张反向的。如果模型很多大或者模型里面有一些奇怪的操作，会把图切开。有可能一个模型生成很多个子图。



