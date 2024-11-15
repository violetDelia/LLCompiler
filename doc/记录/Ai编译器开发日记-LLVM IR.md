# LLVM IR

LLVM IR 是LLVM的一种中间表示，一般我们会将LLVM IR 之前的优化和编译称为前端，LLVM IR 之后的优化和编译成为后端。

使用llc 可以将llvm IR 转为汇编，然后再转为相应后端的汇编指令。

所以LLVM IR是和后端关联性不大的最后一段IR，之后的IR以及优化就和硬件的特性以及指令集强相关了。

## LLVM IR 开头信息

以下是LLVM IR 的开头的部分：

```
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
```

其中source_filename 会注明 IR 的来源，如果是从c文件编译而来，则是c文件的路径，因为IR 是从MLIR 一路翻译而来，默认是LLVMDialectModule。

target datalayout 描述的是机器数据的内存布局，类型大小，对其方式等。

以 `"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128" 为例，`

第一个字母e代表机器数据存储是小端的，如果是E的话就是大端的。

之后的m:`<xx>`代表输出IR的格式。m:e 就是elf 格式的。

px1:x2:x3代表地址空间x1指针大小为x2比特，并且以x3比特对齐。

之后的i64：64 代表整数i64以64比特对齐。fxx：xx同理。

后面的nx ：x：x 表示机器原生支持的整数位宽。

最后的Sxxx  表示栈的对齐大小，S128就代表栈是128比特对齐的。

注：一般对齐方式的格式是i32：32：64，它代表i32类型的最小对齐是32比特，最大对齐是64比特，但是一般最大对齐和最小对齐是一样的，如i32：32：32，可以缩写为i32：32。上面所看到的target都是缩写的形式。

Target triple 则是目标机器是什么，结构是这样的：`arch-verdor-system-env`。如 `x86_64-unknown-linux-gnu`表示机器的arch是x86的，未知供应商，操作系统是linux，环境是gun的。
